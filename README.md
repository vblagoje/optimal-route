# Route Optimizer

A Model Context Protocol (MCP) server that calculates optimal routes between multiple locations using Google Routing Maps APIs.

## MCP Server Capabilities

This server exposes the following MCP primitives:

- **Tools**: 
  - `compute_optimal_route`: Calculate the optimal driving route through specified locations
    - Parameters:
      - `place_names`: List of city or point-of-interest names (3–10 recommended)
      - `alpha`: Weight for travel time in minutes (default: 1.0)
      - `beta`: Weight for travel distance in kilometers (default: 0.2)
      - `round_trip`: Whether to return to the start point i.e. first POI in the list (default: true)

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
docker build -t route-finder .

# Run with environment variables
docker run -p 8080:8080 \
  -e GOOGLE_API_KEY=your_api_key_here \
  -e MCP_TRANSPORT=streamable-http \
  route-finder

# Or using an env file
docker run -p 8080:8080 --env-file .env route-finder
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