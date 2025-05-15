"""
Optimal-route MCP server

Run it with:
    # Default (streamable-http on port 8080)
    python router.py

    # Using SSE transport on port 3000
    python router.py --transport sse --port 3000

    # Using stdio transport
    python router.py --transport stdio

Available transport options:
    - sse
    - streamable-http (default)
    - stdio

Requirements (add to requirements.txt if you haven't already):
    requests
    ortools
    mcp[cli]
"""

import math
import os
from typing import List
import argparse
from urllib.parse import quote_plus

import requests
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("Set GOOGLE_API_KEY in your shell before running.")

# --------------------------------------------------------------------------- #
#  FastMCP server
# --------------------------------------------------------------------------- #

mcp = FastMCP("Route Planner")


# --------------------------------------------------------------------------- #
#  Helper functions (ordinary Python — not visible to MCP)
# --------------------------------------------------------------------------- #


def _geocode(place_names: List[str]) -> list[tuple[float, float]]:
    """
    Resolve place names to (lat, lon) coordinates.

    For accurate geocoding, provide specific place names including city, region, and/or country
    where applicable. For example:
    - "Stratford, Ontario, Canada" instead of just "Stratford"
    - "Stratford, London, UK" instead of just "Stratford"
    - "Pisa, Tuscany, Italy" instead of just "Pisa"

    :param place_names: List of place names to geocode.
    :return: List of (latitude, longitude) tuples.
    :raises ValueError: If geocoding fails or if location is potentially ambiguous.
    """
    coords: list[tuple[float, float]] = []
    for name in place_names:
        url = (
            "https://maps.googleapis.com/maps/api/geocode/json"
            f"?address={name}&key={GOOGLE_API_KEY}"
        )
        data = requests.get(url, timeout=10).json()
        if data["status"] != "OK":
            raise ValueError(f"Geocoding failed for '{name}': {data['status']}")

        # Check for potential ambiguity, help LLM on the next try
        if len(data["results"]) > 1:
            chosen_result = data["results"][0]
            chosen_address = chosen_result["formatted_address"]
            alternative_addresses = [
                r["formatted_address"] for r in data["results"][1:3]
            ]  # Show up to 2 alternatives

            warning = (
                f"Warning: '{name}' is ambiguous and could refer to multiple locations:\n"
                f"- Selected: {chosen_address}\n"
                f"- Alternatives: {', '.join(alternative_addresses)}\n"
                f"To ensure accurate routing, please be more specific (e.g., include city, region, country)."
            )
            raise ValueError(warning)

        loc = data["results"][0]["geometry"]["location"]
        coords.append((loc["lat"], loc["lng"]))
    return coords


def _distance_matrix(coords: list[tuple[float, float]], mode: str) -> list[dict]:
    """Call the Google Distance Matrix (v2) endpoint once and return raw JSON."""
    waypoints = [
        {"waypoint": {"location": {"latLng": {"latitude": lat, "longitude": lon}}}}
        for lat, lon in coords
    ]
    if mode.upper() not in ["DRIVE", "WALK", "BICYCLE", "TRANSIT"]:
        raise ValueError(
            f"Invalid travel mode: {mode}. Valid modes are: DRIVE, WALK, BICYCLE, TRANSIT"
        )
    payload = {
        "travelMode": mode.upper(),
        "origins": waypoints,
        "destinations": waypoints,
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_API_KEY,
        "X-Goog-FieldMask": "originIndex,destinationIndex,duration,distanceMeters",
    }
    url = "https://routes.googleapis.com/distanceMatrix/v2:computeRouteMatrix"
    resp = requests.post(url, headers=headers, json=payload, timeout=25)
    resp.raise_for_status()
    return resp.json()  # type: ignore[no-any-return]


def _parse_matrix(raw: list[dict], n: int) -> tuple[list[list[int]], list[list[float]]]:
    """Convert raw JSON to separate (minutes, km) matrices."""
    minutes = [[0] * n for _ in range(n)]
    km = [[0.0] * n for _ in range(n)]

    for entry in raw:
        i = entry["originIndex"]
        j = entry["destinationIndex"]
        if "duration" in entry:
            minutes[i][j] = int(entry["duration"].rstrip("s")) // 60
        if "distanceMeters" in entry:
            km[i][j] = entry["distanceMeters"] / 1000.0
    return minutes, km


def _cost_matrix(
    mins: list[list[int]],
    km: list[list[float]],
    alpha: float,
    beta: float,
) -> list[list[float]]:
    """alpha·minutes + beta·km."""
    n = len(mins)
    cost = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                cost[i][j] = alpha * mins[i][j] + beta * km[i][j]
    return cost


def _solve_tsp(cost: list[list[float]], round_trip: bool = True) -> list[int]:
    """Solve TSP for given cost matrix. Round trip if round_trip=True, open path otherwise."""
    n = len(cost)
    # The RoutingIndexManager constructor signature differs depending on whether the
    # route starts and ends at the same poi (round-trip) or at distinct pois
    # (open path). For a round-trip, we only need to provide a single poi index. For
    # an open path we must provide lists of start and end indices.
    if round_trip:
        # Single poi (start == end == 0)
        mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
    else:
        # Single poi with fixed start (0) and end (n-1) nodes.  The constructor
        # expects lists for the start and end locations of each poi.
        mgr = pywrapcp.RoutingIndexManager(n, 1, [0], [n - 1])

    routing = pywrapcp.RoutingModel(mgr)
    transit_cb = routing.RegisterTransitCallback(
        lambda i, j: int(cost[mgr.IndexToNode(i)][mgr.IndexToNode(j)])
    )
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )

    sol = routing.SolveWithParameters(params)
    if sol is None:
        raise RuntimeError("TSP solver failed.")

    idx, order = routing.Start(0), []
    while not routing.IsEnd(idx):
        order.append(mgr.IndexToNode(idx))
        idx = sol.Value(routing.NextVar(idx))
    order.append(mgr.IndexToNode(idx))
    return order


def _build_google_maps_url(places: List[str]) -> str:
    """
    Constructs a Google Maps URL for a driving route with up to 25 stops.

    :param places: List of place names in order.
    :return: Google Maps URL as a string.
    """
    if len(places) < 2:
        raise ValueError("At least two places are required to build a route.")
    if len(places) > 25:
        # This limit is based on typical Google Maps URL behavior for multiple waypoints.
        raise ValueError(
            "Google Maps URL generation supports up to 25 stops in a single route."
        )

    base_url = "https://www.google.com/maps/dir/"
    # Ensure all place names are strings, as quote_plus expects strings
    encoded_places = [quote_plus(str(place)) for place in places]
    url = base_url + "/".join(encoded_places)
    return url


# --------------------------------------------------------------------------- #
#  MCP tool (auto-documented from type hints & docstring)
# --------------------------------------------------------------------------- #
@mcp.tool()
def get_distance_matrix(points_of_interest: List[str], mode: str = "DRIVE") -> dict:
    """
    Calculate distance and duration matrices between all pairs of points of interest.

    For accurate results, provide specific place names including city, region, and/or country
    where applicable. For example:
    - "Stratford, Ontario, Canada" instead of just "Stratford"
    - "Stratford, London, UK" instead of just "Stratford"
    - "Pisa, Tuscany, Italy" instead of just "Pisa"

    :param points_of_interest: A list of specific place names to calculate distances between.
    :param mode: Travel mode (DRIVE, WALK, BICYCLE, TRANSIT), defaults to "DRIVE"

    :returns: A dictionary containing:
        - **durations**: Matrix of travel times in minutes between places.
        - **distances**: Matrix of travel distances in kilometers between places.
        - **points_of_interest**: List of points of interest in the same order as matrices.
    """
    coords = _geocode(points_of_interest)
    raw = _distance_matrix(coords, mode=mode)
    mins, kms = _parse_matrix(raw, len(points_of_interest))

    return {
        "durations": mins,
        "durations_unit": "minutes",
        "distances": kms,
        "distances_unit": "km",
        "points_of_interest": points_of_interest,
    }


@mcp.tool()
def compute_optimal_route(
    points_of_interest: List[str],
    alpha: float = 1.0,
    beta: float = 0.2,
    round_trip: bool = True,
    mode: str = "DRIVE",
) -> dict:
    """
    Calculate the optimal route through specified locations.

    For accurate results, provide specific place names including city, region, and/or country
    where applicable. For example:
    - "Stratford, Ontario, Canada" instead of just "Stratford"
    - "Stratford, London, UK" instead of just "Stratford"
    - "Pisa, Tuscany, Italy" instead of just "Pisa"

    :param points_of_interest: A list of specific place names. Must contain between 2 and 25 items, inclusive.
    :param alpha: The weight for travel time in minutes in the cost calculation.
    :param beta: The weight for travel distance in kilometers in the cost calculation.
    :param round_trip: If True (default), return to the first place. If False, end at the last.
    :param mode: Travel mode (DRIVE, WALK, BICYCLE, TRANSIT), defaults to "DRIVE"

    :returns: A dictionary containing:
        - **optimal_route**: List of ordered place names forming the route.
        - **google_maps_url**: A direct link to Google Maps for the generated route.
    """
    num_pois = len(points_of_interest)
    if not (2 <= num_pois <= 25):
        raise ValueError(
            f"The list of points_of_interest must contain between 2 and 25 items. Received {num_pois} items: {points_of_interest}"
        )

    coords = _geocode(points_of_interest)
    raw = _distance_matrix(coords, mode=mode)
    mins, kms = _parse_matrix(raw, len(points_of_interest))
    cost = _cost_matrix(mins, kms, alpha, beta)
    order = _solve_tsp(cost, round_trip)

    ordered_list = [points_of_interest[i] for i in order]
    google_maps_link = _build_google_maps_url(ordered_list)
    return {"optimal_route": ordered_list, "google_maps_url": google_maps_link}


@mcp.tool()
def get_distance_direction(origin: str, destination: str, mode: str = "DRIVE") -> dict:
    """
    Analyzes the comprehensive relationship between two points of interest (POIs),
    including travel distance, estimated travel time, and cardinal direction.

    For accurate results, provide specific place names including city, region, and/or country
    where applicable. For example:
    - "Stratford, Ontario, Canada" instead of just "Stratford"
    - "Stratford, London, UK" instead of just "Stratford"
    - "Pisa, Tuscany, Italy" instead of just "Pisa"

    :param origin: The specific name/address of the starting point of interest.
                  Example: "Pisa, Tuscany, Italy"
    :param destination: The specific name/address of the ending point of interest.
                       Example: "Florence, Tuscany, Italy"
    :param mode: Travel mode (DRIVE, WALK, BICYCLE, TRANSIT), defaults to "DRIVE"

    :returns: A dictionary detailing the relationship, with the following structure:
        - **distance** (dict): Contains travel distance information.
            - **value** (float): The numerical value of the distance.
            - **unit** (str): The unit of distance (e.g., "km").
        - **duration** (dict): Contains estimated travel time information.
            - **value** (int): The numerical value of the travel time.
            - **unit** (str): The unit of time (e.g., "minutes").
            - **mode** (str): The travel mode used for this estimation.
        - **cardinal** (dict): Contains cardinal direction and coordinate information.
            - **origin** (dict): Details of the origin POI.
                - **name** (str): The provided name/address of the origin POI.
                - **lat** (float): Latitude of the geocoded origin.
                - **lon** (float): Longitude of the geocoded origin.
            - **destination** (dict): Details of the destination POI.
                - **name** (str): The provided name/address of the destination POI.
                - **lat** (float): Latitude of the geocoded destination.
                - **lon** (float): Longitude of the geocoded destination.
            - **bearing** (float): Compass bearing from origin to destination, in degrees (0-360).
            - **direction** (str): Standard two-letter cardinal direction
                                         (e.g., "N", "NE", "E", "SE", "S", "SW", "W", "NW").
        - **explanation** (str): An LLM-friendly, human-readable summary combining distance,
                               duration, and directional information.
    """
    # Get coordinates for both POIs
    coords = _geocode([origin, destination])
    if not coords or len(coords) != 2:
        return {
            "distance": None,
            "duration": None,
            "cardinal": {},
            "explanation": "Failed to geocode one or both points of interest.",
        }

    # Get distance and duration
    raw = _distance_matrix(coords, mode=mode)
    mins, kms = _parse_matrix(raw, 2)
    distance = kms[0][1]
    duration = mins[0][1]

    # Extract coordinates
    origin_lat, origin_lon = coords[0]
    dest_lat, dest_lon = coords[1]

    # Determine cardinal relationships
    is_north = dest_lat > origin_lat
    is_south = dest_lat < origin_lat
    is_east = dest_lon > origin_lon
    is_west = dest_lon < origin_lon

    # Determine cardinal direction code (e.g., N, NE, E, etc.)
    if is_north and is_east:
        cardinal_direction = "NE"
    elif is_north and is_west:
        cardinal_direction = "NW"
    elif is_south and is_east:
        cardinal_direction = "SE"
    elif is_south and is_west:
        cardinal_direction = "SW"
    elif is_north:
        cardinal_direction = "N"
    elif is_south:
        cardinal_direction = "S"
    elif is_east:
        cardinal_direction = "E"
    elif is_west:
        cardinal_direction = "W"
    else:
        raise ValueError("Failed to determine cardinal direction.")

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(origin_lat)
    lon1 = math.radians(origin_lon)
    lat2 = math.radians(dest_lat)
    lon2 = math.radians(dest_lon)

    # Calculate bearing
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        dlon
    )
    bearing = math.atan2(y, x)

    # Convert bearing from radians to degrees
    bearing = math.degrees(bearing)
    # Normalize bearing to 0-360 degrees
    bearing = (bearing + 360) % 360

    # Create comprehensive explanation
    explanation = f"{destination} is {cardinal_direction} of {origin} "
    explanation += f"at bearing {bearing:.1f}°. "
    explanation += (
        f"It is {distance:.1f} km away ({duration} minutes by {mode.lower()}). "
    )
    explanation += f"Coordinates: {origin} ({origin_lat:.6f}, {origin_lon:.6f}), "
    explanation += f"{destination} ({dest_lat:.6f}, {dest_lon:.6f})"

    return {
        "distance": {
            "value": distance,
            "unit": "km",
        },
        "duration": {
            "value": duration,
            "unit": "minutes",
            "mode": mode,
        },
        "cardinal": {
            "origin": {
                "name": origin,
                "lat": origin_lat,
                "lon": origin_lon,
            },
            "destination": {
                "name": destination,
                "lat": dest_lat,
                "lon": dest_lon,
            },
            "bearing": bearing,
            "direction": cardinal_direction,
        },
        "explanation": explanation,
    }


# --------------------------------------------------------------------------- #
#  Entrypoint
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MCP server.")
    parser.add_argument(
        "--transport",
        choices=["sse", "streamable-http", "stdio"],
        default="streamable-http",
        help="Transport method to use (default: streamable-http)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind the server to (default: 8080)",
    )
    args = parser.parse_args()

    mcp.settings.host = args.host
    mcp.settings.port = args.port
    mcp.run(
        transport=args.transport,
    )
