import io

import pytest
from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.api import deps
from app.api.main import app
from app.config import Settings
from app.embeddings.hashing import HashingEmbedder

SOLAR = ("Solar panels convert sunlight into electricity using photovoltaic cells. "
         "Residential solar installations typically cost between 15,000 and 25,000 dollars before incentives. "
         "The payback period for rooftop solar is usually seven to ten years.")
WIND = ("Wind turbines generate electricity from moving air. Onshore wind farms have lower capital cost per "
        "megawatt than offshore wind. Turbine noise and visual impact are common community concerns.")
COOK = "Sourdough bread requires a starter, flour, water and salt. Fermentation takes twelve hours at room temperature."


def make_pdf(pages: list[str]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in pages:
        y = 750
        words, line = text.split(), ""
        for w in words:
            if len(line) + len(w) > 80:
                c.drawString(50, y, line); y -= 14; line = ""
            line += w + " "
        c.drawString(50, y, line)
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.fixture
def settings():
    return Settings(embedding_backend="hashing", chunk_size=300, chunk_overlap=40, min_score=0.05)


@pytest.fixture
def services(settings):
    return deps.build_services(settings, embedder=HashingEmbedder(), persist=False)


@pytest.fixture
def client(services):
    app.dependency_overrides[deps.get_services] = lambda: services
    yield TestClient(app)
    app.dependency_overrides.clear()
