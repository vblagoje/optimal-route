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

import os
from typing import List
import argparse

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
    """Resolve place names to (lat, lon)."""
    coords: list[tuple[float, float]] = []
    for name in place_names:
        url = (
            "https://maps.googleapis.com/maps/api/geocode/json"
            f"?address={name}&key={GOOGLE_API_KEY}"
        )
        data = requests.get(url, timeout=10).json()
        if data["status"] != "OK":
            raise ValueError(f"Geocoding failed for '{name}': {data['status']}")
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
    if round_trip:
        mgr = pywrapcp.RoutingIndexManager(n, 1, 0)
    else:
        mgr = pywrapcp.RoutingIndexManager(n, 1, 0, n - 1)

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


# --------------------------------------------------------------------------- #
#  MCP tool (auto-documented from type hints & docstring)
# --------------------------------------------------------------------------- #
@mcp.tool()
def get_distance_matrix(points_of_interest: List[str], mode: str = "DRIVE") -> dict:
    """
    Calculate distance and duration matrices between all pairs of points of interest.

    :param points_of_interest: A list of points of interest to calculate distances between.
    :param mode: Travel mode (DRIVE, WALK, BICYCLE, TRANSIT)

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
def get_distance(origin: str, destination: str, mode: str = "DRIVE") -> dict:
    """
    Calculate distance and duration between two points of interest.

    :param origin: Starting point of interest.
    :param destination: Ending point of interest.
    :param mode: Travel mode (DRIVE, WALK, BICYCLE, TRANSIT)

    :returns: A dictionary containing:
        - **duration**: Travel time in minutes.
        - **duration_unit**: "minutes"
        - **distance**: Travel distance in kilometers.
        - **distance_unit**: "km"
        - **origin**: Origin point of interest.
        - **destination**: Destination point of interest.
    """
    coords = _geocode([origin, destination])
    raw = _distance_matrix(coords, mode=mode)
    mins, kms = _parse_matrix(raw, 2)

    return {
        "duration": mins[0][1],
        "duration_unit": "minutes",
        "distance": kms[0][1],
        "distance_unit": "km",
        "origin": origin,
        "destination": destination,
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

    :param points_of_interest: A list of city or point-of-interest names (3–10 recommended).
    :param alpha: The weight for travel time in minutes in the cost calculation.
    :param beta: The weight for travel distance in kilometers in the cost calculation.
    :param round_trip: If True (default), return to the first place. If False, end at the last.
    :param mode: Travel mode (DRIVE, WALK, BICYCLE, TRANSIT)

    :returns: A dictionary containing:
        - **optimal_route**: List of ordered place names forming the route.
    """
    if len(points_of_interest) < 2:
        raise ValueError("Provide at least two points of interest.")

    coords = _geocode(points_of_interest)
    raw = _distance_matrix(coords, mode=mode)
    mins, kms = _parse_matrix(raw, len(points_of_interest))
    cost = _cost_matrix(mins, kms, alpha, beta)
    order = _solve_tsp(cost, round_trip)

    ordered_list = [points_of_interest[i] for i in order]
    return {"optimal_route": ordered_list}


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
