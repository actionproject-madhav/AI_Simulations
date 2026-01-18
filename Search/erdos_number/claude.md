# Wikipedia Chain Finder

## Project Summary
A web app that finds the shortest path of links between two Wikipedia articles using bidirectional search. Flask backend with vanilla JS frontend.

## Core Requirements
- User enters start/end article titles, app finds the shortest link path connecting them.
- Maximum search depth: 7 links total.
- Uses Wikipedia MediaWiki API (no local database for the full link map, only caching).
- Only mainspace articles (namespace 0).
- Bidirectional iterative deepening search algorithm.

## Architecture
- **Backend**: Python/Flask with threading for background jobs.
- **Frontend**: Vanilla HTML/CSS/JS with "Rich Aesthetics".
- **Database**: SQLite for link caching.
- **Communication**: Polling pattern—POST creates job, GET polls for status.

## Coding Standards
- Polite User-Agent for Wikipedia API.
- Error handling for API pagination and missing pages.
- Standard Python type hints and docstrings.
