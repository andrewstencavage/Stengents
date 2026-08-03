"""A local stand-in for strava.com's training-log page, used only to validate
``strava_playwright.list_recent_activities`` end to end without real Strava
credentials."""

from .server import serve

__all__ = ["serve"]
