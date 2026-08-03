"""A local stand-in for strava.com's upload flow, used only to validate
``strava_playwright.upload_fit`` end to end without real Strava credentials."""

from .server import serve

__all__ = ["serve"]
