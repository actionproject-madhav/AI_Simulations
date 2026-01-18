# Specs: Six Degrees of Wikipedia

## Goal
Build a web application that finds the shortest path of links between two Wikipedia articles using bidirectional iterative deepening search.

## Features
- **Bidirectional Iterative Deepening Search**: Search starts from both ends to meet in the middle.
- **Wikipedia API**: Dynamically fetch page content and parse links (Namespace 0 only).
- **Polling Backend**: Flask API that manages long-running search jobs.
- **Vanilla Frontend**: Simple UI with two input boxes for titles.
- **SQLite Caching**: Permanent storage for page links to optimize repeated searches.
- **Depth Limit**: Failure if no path is found within 7 links (max search depth 4 from each side).

## Technical Requirements
- **Backend**: Python (Flask, requests).
- **Frontend**: HTML, CSS, Vanilla JS.
- **Database**: SQLite.
- **Platform Compatibility**: Designed to be deployed on Vercel.
