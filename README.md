# Optimal Route MCP Server

A Model Context Protocol (MCP) server for calculating optimal travel routes and distances between locations using the Google Maps Platform.

## 🧰 Tools

This server exposes the following MCP tools:

### `compute_optimal_route`

Returns the optimal path through a list of points of interest.

**Parameters:**

* `points_of_interest` (list of str): List of locations (3–10 recommended).
* `alpha` (float): Weight for travel time (default: `1.0`).
* `beta` (float): Weight for distance (default: `0.2`).
* `round_trip` (bool): Whether to return to starting point (default: `true`).
* `mode` (str): Travel mode (`DRIVE`, `WALK`, `BICYCLE`, `TRANSIT`; default: `DRIVE`).

**Returns:**

* `optimal_route` (list of str): Ordered place names forming the best route.

---

### `get_distance_matrix`

Returns time and distance matrices between all locations.

**Parameters:**

* `points_of_interest` (list of str)
* `mode` (str): Travel mode (default: `DRIVE`)

**Returns:**

* `durations` (list\[list\[float]]): Travel times (in minutes)
* `distances` (list\[list\[float]]): Travel distances (in km)
* `points_of_interest` (list\[str])
* `durations_unit`: `"minutes"`
* `distances_unit`: `"km"`

---

### `get_distance`

Returns time and distance between two locations.

**Parameters:**

* `origin` (str)
* `destination` (str)
* `mode` (str): Travel mode (default: `DRIVE`)

**Returns:**

* `duration` (float)
* `distance` (float)
* `origin` (str)
* `destination` (str)
* `duration_unit`: `"minutes"`
* `distance_unit`: `"km"`

---

## 🔧 Environment

**Required:**

* `GOOGLE_API_KEY`: Google Maps API key (with Geocoding, Routes, Distance Matrix access)

**Optional:**

* `MCP_TRANSPORT`: One of `sse`, `streamable-http`, or `stdio` (default: `sse`)

---

## 🐳 Docker

```bash
# Build
docker build -t vblagoje/optimal-route .

# Run
docker run -p 8080:8080 \
  -e GOOGLE_API_KEY=your_key_here \
  -e MCP_TRANSPORT=streamable-http \
  vblagoje/optimal-route
```

Or use an `.env` file:

```bash
docker run -p 8080:8080 --env-file .env vblagoje/optimal-route
```

`.env` example:

```env
GOOGLE_API_KEY=your_key_here
MCP_TRANSPORT=streamable-http
```

---

## 🧪 Usage Example (Python SDK)

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async def main():
    async with streamablehttp_client("http://localhost:8080/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "compute_optimal_route",
                {
                    "points_of_interest": [
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

Make sure the server is using `streamable-http` transport for this to work.
