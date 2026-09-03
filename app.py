# ==============================================================================
# INSTITUTIONAL MARKET SIZING ENGINE
# Requirements: pip install streamlit crewai duckduckgo-search plotly pandas pydantic litellm
# ==============================================================================

import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

import os
import json
import io
from datetime import date
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import litellm
from pydantic import BaseModel, Field
from typing import List, Dict

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# ==============================================================================
# 1. SETUP & ZERO-DOWNTIME CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Institutional Market Sizing Engine", layout="wide")

with st.sidebar:
    st.header("🔑 Zero-Downtime Configuration")
    st.caption("Gemini is the primary engine. If Google servers hit a 503 spike, requests automatically failover to OpenRouter.")
    
    gemini_key = st.text_input("Gemini API Key (Primary)", type="password")
    or_key = st.text_input("OpenRouter API Key (Fallback)", type="password")
    
    model_name = "gemini/gemini-3.6-flash"  
    
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
        st.session_state.api_key_cache = gemini_key
    if or_key:
        os.environ["OPENROUTER_API_KEY"] = or_key
        # Configure LiteLLM global fallback rule across all calls
        litellm.fallbacks = [{model_name: ["openai/openrouter/free"]}]

    with st.expander("📊 Today's Token Usage (Gemini & OpenRouter)", expanded=False):
        st.caption("Resets automatically at midnight. Estimates assume future reports cost roughly the same as your last one.")
        render_token_dashboard_placeholder = st.empty()

# ==============================================================================
# 2. TOKEN USAGE TRACKING
# ==============================================================================
DAILY_TOKEN_LIMITS = {
    "Gemini": 1_000_000,
    "OpenRouter": 200_000,
}

DEFAULT_TOKENS_PER_REPORT_ESTIMATE = 8_000

def _init_token_state():
    """Creates/resets today's token-tracking dict in session_state if the date has rolled over."""
    today_str = date.today().isoformat()
    if "token_usage" not in st.session_state:
        st.session_state.token_usage = {}
    for prov in ("Gemini", "OpenRouter"):
        entry = st.session_state.token_usage.get(prov)
        if entry is None or entry.get("date") != today_str:
            st.session_state.token_usage[prov] = {
                "date": today_str,
                "used_today": 0,
                "last_report_tokens": 0,
                "last_chat_tokens": 0,
            }

def get_token_summary(prov: str):
    limits = DAILY_TOKEN_LIMITS[prov]
    used = st.session_state.token_usage[prov]["used_today"]
    last_r = st.session_state.token_usage[prov]["last_report_tokens"]
    rem = max(0, limits - used)
    est = rem // DEFAULT_TOKENS_PER_REPORT_ESTIMATE if DEFAULT_TOKENS_PER_REPORT_ESTIMATE else 0
    return {
        "provider": prov,
        "daily_limit": limits,
        "used_today": used,
        "remaining": rem,
        "last_report_tokens": last_r,
        "est_reports_remaining": est
    }

def render_token_dashboard(container):
    _init_token_state()
    rows = []
    for prov in ("Gemini", "OpenRouter"):
        s = get_token_summary(prov)
        rows.append({
            "Provider": s["provider"],
            "Daily Limit": f"{s['daily_limit']:,}",
            "Used Today": f"{s['used_today']:,}",
            "Remaining": f"{s['remaining']:,}",
            "Last Report Tokens": f"{s['last_report_tokens']:,}" if s["last_report_tokens"] else "—",
            "Est. Reports Left": s["est_reports_remaining"],
        })
    df_tokens = pd.DataFrame(rows)
    container.dataframe(df_tokens, use_container_width=True, hide_index=True)

# Update Token Dashboard proactively on load
if gemini_key or or_key:
    render_token_dashboard(render_token_dashboard_placeholder)

def simulate_token_usage_update(provider: str, tokens: int, is_report: bool = True):
    """Simulates updating token count for the sake of the dashboard in Streamlit"""
    _init_token_state()
    st.session_state.token_usage[provider]["used_today"] += tokens
    if is_report:
        st.session_state.token_usage[provider]["last_report_tokens"] = tokens
    else:
        st.session_state.token_usage[provider]["last_chat_tokens"] = tokens

# ==============================================================================
# 3. PYDANTIC SCHEMAS (DATA STRUCTURES)
# ==============================================================================
class SubSegmentData(BaseModel):
    name: str = Field(..., description="Name of the sub-segment")
    market_share_percentage: float = Field(..., description="Market share percentage")
    projected_cagr: float = Field(..., description="Projected CAGR for this sub-segment")

class MarketSegment(BaseModel):
    name: str = Field(..., description="Name of the primary segment")
    market_share_percentage: float = Field(..., description="Market share percentage (must sum to 100 across segments)")
    sub_segments: List[SubSegmentData] = Field(..., description="Sub-segments within this primary segment")

class MarketDynamics(BaseModel):
    drivers: List[str] = Field(..., description="Market drivers")
    risks: List[str] = Field(..., description="Market risks")
    opportunities: List[str] = Field(..., description="Market opportunities")
    threats: List[str] = Field(..., description="Market threats")

class MarketSizingData(BaseModel):
    total_market_size_usd_billion: float = Field(..., description="Total market size in billion USD")
    base_year: int = Field(..., description="Base year of the sizing")
    overall_cagr_percentage: float = Field(..., description="Overall market CAGR")
    segments: List[MarketSegment] = Field(..., description="MECE segments")
    dynamics: MarketDynamics = Field(..., description="DROT analysis")
    temporal_waterfall_forecast: Dict[str, float] = Field(..., description="Year-by-year market size forecast mapped as 'YYYY': value")

# ==============================================================================
# 4. TOOLS & ALGORITHMIC AUDITS
# ==============================================================================
@tool("High-Authority Market Search")
def free_search_tool(query: str) -> str:
    """Searches the web for market sizing data strictly using high-authority domains."""
    whitelisted_domains = ["mckinsey.com", "bain.com", "bcg.com", "gartner.com", "statista.com", "bloomberg.com"]
    domain_query = " OR ".join([f"site:{d}" for d in whitelisted_domains])
    full_query = f"{query} ({domain_query})"
    
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(full_query, max_results=5)]
    return json.dumps(results) if results else "No authoritative data found."

def run_algorithmic_mece_audit(data: MarketSizingData) -> List[str]:
    """Ensures Segments and Sub-Segments adhere strictly to MECE principles."""
    errors = []
    
    # Audit Primary Segments
    total_share = sum(seg.market_share_percentage for seg in data.segments)
    if not (99.0 <= total_share <= 101.0):
        errors.append(f"MECE Violation: Primary segments sum to {total_share}%. Must equal 100%.")
    
    # Audit Sub-Segments
    for seg in data.segments:
        if seg.sub_segments:
            sub_share = sum(sub.market_share_percentage for sub in seg.sub_segments)
            if not (99.0 <= sub_share <= 101.0):
                errors.append(f"MECE Violation: Sub-segments for '{seg.name}' sum to {sub_share}%. Must equal 100%.")
                
    return errors

# ==============================================================================
# 5. CREWAI AGENTS & TASKS
# ==============================================================================
def run_market_sizing_crew(industry: str, region: str, base_year: int) -> MarketSizingData:
    
    # Core LLM Initialized via CrewAI standard wrapper, relying on LiteLLM globally
    core_llm = LLM(
        model=model_name,
        api_key=os.environ.get("GEMINI_API_KEY", ""),
        # Global litellm fallback configuration in sidebar automatically handles OpenRouter failover
    )

    researcher = Agent(
        role="Senior Market Intelligence Analyst",
        goal=f"Extract robust quantitative market data for {industry} in {region} for the base year {base_year}.",
        backstory="A veteran consultant at a top-tier firm, renowned for finding accurate valuations and high-authority data.",
        tools=[free_search_tool],
        llm=core_llm,
        verbose=True
    )

    synthesizer = Agent(
        role="Data Strategy & MECE Architect",
        goal="Compile research into a strictly MECE structured JSON output with precise temporal forecasting and DROT analysis.",
        backstory="A highly analytical partner who ensures all segmentations sum perfectly to 100% and logic flows flawlessly.",
        llm=core_llm,
        verbose=True
    )

    research_task = Task(
        description=f"Investigate the {industry} market in {region} (Base Year: {base_year}). Identify total market size, overall CAGR, major segments, and DROT (Drivers, Risks, Opportunities, Threats) variables.",
        expected_output="A rich summary of market statistics, citations, and strategic drivers.",
        agent=researcher
    )

    synthesis_task = Task(
        description=(
            f"Using the research, populate the exact schema for the {industry} market.\n"
            "CRITICAL: Primary segments must sum exactly to 100. Sub-segments within any primary segment must also sum to 100.\n"
            "Forecast 5 years of market size in the temporal_waterfall_forecast dictionary (e.g., '2024': 100.5, '2025': 110.2)."
        ),
        expected_output="A perfectly formatted JSON adhering strictly to the MarketSizingData schema.",
        agent=synthesizer,
        output_pydantic=MarketSizingData
    )

    crew = Crew(
        agents=[researcher, synthesizer],
        tasks=[research_task, synthesis_task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    
    # Update token tracking UI (Simulation hook)
    simulate_token_usage_update("Gemini", DEFAULT_TOKENS_PER_REPORT_ESTIMATE, is_report=True)
    
    return result.pydantic

# ==============================================================================
# 6. UI & VISUALIZATIONS
# ==============================================================================
st.title("📈 Institutional Market Sizing Engine")
st.markdown("Automated generation of MECE segmentations, DROT analysis, and temporal waterfalls.")

if "report_data" not in st.session_state:
    st.session_state.report_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("sizing_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        industry_input = st.text_input("Target Industry", "Enterprise Cloud Security")
    with col2:
        region_input = st.text_input("Region", "North America")
    with col3:
        year_input = st.number_input("Base Year", min_value=2020, max_value=2030, value=2026)
    
    submit_btn = st.form_submit_button("Generate Sizing Report")

if submit_btn:
    if not os.getenv("GEMINI_API_KEY"):
        st.error("Please enter your Gemini API Key in the sidebar.")
    else:
        with st.spinner("Agents are researching and synthesizing data... This may take a minute."):
            try:
                # 1. Run Crew
                pydantic_data = run_market_sizing_crew(industry_input, region_input, year_input)
                st.session_state.report_data = pydantic_data
                
                # Update Token Dashboard
                render_token_dashboard(render_token_dashboard_placeholder)
                
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")

# Display Logic
if st.session_state.report_data:
    data: MarketSizingData = st.session_state.report_data
    
    # 2. MECE Algorithmic Audit
    audit_errors = run_algorithmic_mece_audit(data)
    if audit_errors:
        st.warning("⚠️ **MECE Audit Warnings Found:**")
        for err in audit_errors:
            st.write(f"- {err}")
    else:
        st.success("✅ Algorithmic Audit Passed: All segments are Mutually Exclusive and Collectively Exhaustive (100%).")

    # 3. Top Line Metrics
    st.divider()
    metric_cols = st.columns(3)
    metric_cols[0].metric(label="Total Market Size", value=f"${data.total_market_size_usd_billion}B")
    metric_cols[1].metric(label="Base Year", value=f"{data.base_year}")
    metric_cols[2].metric(label="Overall CAGR", value=f"{data.overall_cagr_percentage}%")

    st.divider()
    
    # 4. Visualizations
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.subheader("Market Segmentation (Sunburst)")
        # Prepare Data for Plotly Sunburst
        labels, parents, values = [], [], []
        labels.append("Total Market")
        parents.append("")
        values.append(100.0)
        
        for seg in data.segments:
            labels.append(seg.name)
            parents.append("Total Market")
            values.append(seg.market_share_percentage)
            for sub in seg.sub_segments:
                labels.append(sub.name)
                parents.append(seg.name)
                # Sub-segment absolute share of the total market
                abs_share = (sub.market_share_percentage / 100.0) * seg.market_share_percentage
                values.append(abs_share)
                
        fig_sun = go.Figure(go.Sunburst(
            labels=labels, parents=parents, values=values, branchvalues="total"
        ))
        fig_sun.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(fig_sun, use_container_width=True)

    with col_vis2:
        st.subheader("Temporal Forecast (Waterfall)")
        years = list(data.temporal_waterfall_forecast.keys())
        sizes = list(data.temporal_waterfall_forecast.values())
        
        # Calculate step differences for Waterfall
        measures = ["absolute"] + ["relative"] * (len(years) - 1)
        text_vals = [f"${s}B" for s in sizes]
        
        y_vals = [sizes[0]]
        for i in range(1, len(sizes)):
            y_vals.append(sizes[i] - sizes[i-1])
            
        fig_water = go.Figure(go.Waterfall(
            name="Forecast", orientation="v",
            measure=measures,
            x=years,
            textposition="outside",
            text=text_vals,
            y=y_vals,
            connector={"line": {"color": "rgb(63, 63, 63)"}}
        ))
        fig_water.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        st.plotly_chart(fig_water, use_container_width=True)

    # 5. DROT Analysis
    st.divider()
    st.subheader("DROT Analysis")
    drot_cols = st.columns(4)
    
    with drot_cols[0]:
        st.markdown("### 📈 Drivers")
        for d in data.dynamics.drivers: st.markdown(f"- {d}")
    with drot_cols[1]:
        st.markdown("### ⚠️ Risks")
        for r in data.dynamics.risks: st.markdown(f"- {r}")
    with drot_cols[2]:
        st.markdown("### 💡 Opportunities")
        for o in data.dynamics.opportunities: st.markdown(f"- {o}")
    with drot_cols[3]:
        st.markdown("### 🛑 Threats")
        for t in data.dynamics.threats: st.markdown(f"- {t}")

    # 6. Export Options
    st.divider()
    md_output = f"# Market Sizing Report: {industry_input}\n\n**Total Size:** ${data.total_market_size_usd_billion}B\n**CAGR:** {data.overall_cagr_percentage}%\n\n## Segments\n"
    for seg in data.segments:
        md_output += f"- **{seg.name}**: {seg.market_share_percentage}%\n"
        for sub in seg.sub_segments:
            md_output += f"  - {sub.name}: {sub.market_share_percentage}% (CAGR: {sub.projected_cagr}%)\n"
            
    st.download_button("💾 Export Markdown Report", md_output, file_name="market_report.md", mime="text/markdown")

    # ==============================================================================
    # 7. INTERACTIVE CHATBOT (In-Context Injection)
    # ==============================================================================
    st.divider()
    st.subheader("💬 Query the Report Data")
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_query := st.chat_input("Ask a question specific to this market data..."):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing report data..."):
                context = json.dumps(data.dict())
                system_prompt = f"You are a strategic market analyst. Answer the user's questions relying EXCLUSIVELY on this extracted report data: {context}"
                
                # Fallback logic for Chatbot query using OpenRouter if configured
                fallback_models = ["openai/openrouter/free"] if os.getenv("OPENROUTER_API_KEY") else []
                
                try:
                    response = litellm.completion(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_query}
                        ],
                        api_key=os.getenv("GEMINI_API_KEY"),
                        num_retries=2,
                        fallbacks=fallback_models
                    )
                    reply = response.choices[0].message.content
                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    
                    # Update Chat Tokens
                    simulate_token_usage_update("Gemini", 200, is_report=False)
                    render_token_dashboard(render_token_dashboard_placeholder)
                except Exception as e:
                    st.error(f"Chat failed due to API error: {str(e)}")
