from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import COLORS


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"


def load_css() -> None:
    css = (ASSET_DIR / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def hero_art() -> str:
    return (ASSET_DIR / "hero.svg").read_text(encoding="utf-8")


def stat_card(label: str, value: str, tone: str) -> None:
    color = COLORS[tone]
    st.markdown(
        f"""
        <div class="stat-card" style="background:{color}">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, body: str | None = None) -> None:
    html = f'<div class="section-kicker">{kicker}</div><h2>{title}</h2>'
    if body:
        html += f'<p class="section-copy">{body}</p>'
    st.markdown(html, unsafe_allow_html=True)


def result_block(label: str, value: str, detail: str, tone: str = "lime") -> None:
    color = COLORS[tone]
    st.markdown(
        f"""
        <div class="result-block" style="background:{color}">
            <div class="result-label">{label}</div>
            <div class="result-value">{value}</div>
            <div class="result-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
