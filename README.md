# Route Optimizer

A Model Context Protocol (MCP) server that calculates optimal routes between multiple locations using Google Routing Maps APIs.

## MCP Server Capabilities

This server exposes the following MCP tools:

- `compute_optimal_route`: Calculate the optimal route through specified locations.
    - Parameters:
      - `points_of_interest`: A list of city or point-of-interest names (3–10 recommended).
      - `alpha`: The weight for travel time in minutes in the cost calculation (default: 1.0).
      - `beta`: The weight for travel distance in kilometers in the cost calculation (default: 0.2).
      - `round_trip`: If True (default), return to the first place. If False, end at the last.
      - `mode`: Travel mode (default: `DRIVE`). Available options:
        - `DRIVE`: Driving mode.
        - `WALK`: Walking mode.
        - `BICYCLE`: Cycling mode.
        - `TRANSIT`: Public transit mode.
    - Returns:
      - `optimal_route`: List of ordered place names forming the route.
 - `get_distance_matrix`: Calculate distance and duration matrices between all pairs of points of interest.
    - Parameters:
      - `points_of_interest`: A list of points of interest to calculate distances between.
      - `mode`: Travel mode (default: `DRIVE`). Available options:
        - `DRIVE`: Driving mode.
        - `WALK`: Walking mode.
        - `BICYCLE`: Cycling mode.
        - `TRANSIT`: Public transit mode.
    - Returns:
      - `durations`: Matrix of travel times in minutes between places.
      - `durations_unit`: Unit for durations, which is "minutes".
      - `distances`: Matrix of travel distances in kilometers between places.
      - `distances_unit`: Unit for distances, which is "km".
      - `points_of_interest`: List of points of interest in the same order as matrices.

 - `get_distance`: Calculate distance and duration between two points of interest.
    - Parameters:
      - `origin`: Starting point of interest.
      - `destination`: Ending point of interest.
      - `mode`: Travel mode (default: `DRIVE`). Available options:
        - `DRIVE`: Driving mode.
        - `WALK`: Walking mode.
        - `BICYCLE`: Cycling mode.
        - `TRANSIT`: Public transit mode.
    - Returns:
      - `duration`: Travel time in minutes.
      - `duration_unit`: Unit for duration, which is "minutes".
      - `distance`: Travel distance in kilometers.
      - `distance_unit`: Unit for distance, which is "km".
      - `origin`: Origin point of interest.
      - `destination`: Destination point of interest. 
  
## Environment Variables

The following environment variables are required:

- `GOOGLE_API_KEY`: Your Google Maps API key with access to:
  - Geocoding API
  - Distance Matrix API
  - Routes API

Optional environment variables:

- `MCP_TRANSPORT`: Transport method to use (default: "sse")
  - Available options:
    - `sse`: Server-Sent Events transport
    - `streamable-http`: Streamable HTTP transport
    - `stdio`: Standard IO transport

## Running with Docker

```bash
# Build the container
docker build -t vblagoje/optimal-route .

# Run with environment variables
docker run -p 8080:8080 \
  -e GOOGLE_API_KEY=your_api_key_here \
  -e MCP_TRANSPORT=streamable-http \
  vblagoje/optimal-route

# Or using an env file
docker run -p 8080:8080 --env-file .env optimal-route
```

## Example .env file
```
GOOGLE_API_KEY=your_api_key_here
MCP_TRANSPORT=streamable-http
```

## Interacting with the Server

The server can be accessed using any MCP client. For example, using the MCP Python SDK:

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://localhost:8080/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Calculate route through European cities
            result = await session.call_tool(
                "compute_optimal_route",
                {
                    "place_names": [
                        "Paris, France",
                        "Berlin, Germany",
                        "Rome, Italy",
                        "Amsterdam, Netherlands"
                    ],
                    "round_trip": True
                }
            )
            print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

Note: in the above script pay attention which transport is the server using, the above example assumes streamable-http
