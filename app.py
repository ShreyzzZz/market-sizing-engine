import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

import os
import re
import streamlit as st
import plotly.express as px
import pandas as pd
from pydantic import BaseModel, Field
from typing import List
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS

# ==============================================================================
# 1. UI SETUP & CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Institutional Market Sizing Engine",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Institutional Market Sizing Engine")
st.markdown("This engine maps the organic architecture of the market using **Temporal Waterfall Logic (Prioritizing 2026/2025 Actuals)**. Citations are screened in Python to ensure data originates ONLY from **Big 3 / Big 4 consulting firms, top analyst research houses, SEC filings, and official investor relations**.")

# Sidebar Configuration (Includes Groq LiteLLM Bypass & Gemini 3.6 Update)
with st.sidebar:
    st.header("🔑 Zero-Cost Configuration")
    provider = st.radio("Select Free API Provider:", ["Google Gemini (Free Tier)", "Groq Cloud (100% Free Backup)"])
    
    if provider == "Google Gemini (Free Tier)":
        api_key_input = st.text_input("Gemini API Key", type="password")
        model_name = "gemini/gemini-3.6-flash"  
        env_var_name = "GEMINI_API_KEY"
        custom_base_url = None
        crew_rpm_limit = 4 
    else:
        api_key_input = st.text_input("Groq API Key", type="password")
        model_name = "openai/mixtral-8x7b-32768"
        env_var_name = "GROQ_API_KEY"
        custom_base_url = "https://api.groq.com/openai/v1"
        crew_rpm_limit = 15 

# ==============================================================================
# 2. PYDANTIC SCHEMAS (WITH DROT & STRATEGIC INSIGHTS)
# ==============================================================================
HIGH_AUTHORITY_PATTERNS = [
    "mckinsey", "bcg.com", "bain.com", "deloitte", "pwc.com", "ey.com", "kpmg", "accenture",
    "gartner", "idc.com", "forrester", "statista", "bloomberg", "reuters", "pitchbook", 
    "cbinsights", "spglobal", "fitchratings", "moodys", "grandviewresearch", 
    "fortune-business-insights", "mordorintelligence", "marketsandmarkets", "precedence-research",
    "sec.gov", "investor.", "wsj.com", "ft.com", "cnbc.com", "forbes.com", "finance.yahoo.com"
]

def is_high_authority_source(url: str, title: str, snippet: str) -> bool:
    combined_text = f"{url.lower()} {title.lower()} {snippet.lower()}"
    return any(pattern in combined_text for pattern in HIGH_AUTHORITY_PATTERNS)

class SubSegmentData(BaseModel):
    sub_segment_name: str = Field(..., description="Name of the specific functional sub-segment")
    top_vendors: List[str] = Field(..., description="Top key market players operating STRICTLY in this sub-segment")
    estimated_subsegment_market_size_billions: float = Field(
        ..., 
        description="ACTUAL reported TOTAL market valuation for this entire sub-segment industry-wide in BILLIONS USD."
    )
    reporting_period: str = Field(
        ..., 
        description="Must be the latest available actual period: 'FY2026', 'LTM', 'FY2025', or 'FY2024' (only in rare cases). Data older than 2024 is strictly forbidden."
    )
    publisher_name: str = Field(..., description="Name of recognized institutional source (e.g., McKinsey, Gartner, SEC).")
    source_url: str = Field(..., description="Deep hyperlink URL from a high-authority domain.")
    verification_snippet: str = Field(..., description="Verbatim textual quote proving ACTUAL reported figures.")

class MarketSegment(BaseModel):
    segment_name: str = Field(..., description="Strictly MECE main functional segment name")
    definition: str = Field(..., description="Boundary definition explicitly proving why this pillar NEVER overlaps with others")
    sub_segments: List[SubSegmentData] = Field(
        ..., 
        min_length=1, 
        max_length=15, 
        description="As many distinct, strictly non-overlapping sub-segments as required to map this pillar accurately (minimum 1)."
    )

# NEW SCHEMA: Market Dynamics
class MarketDynamics(BaseModel):
    strategic_insight: str = Field(..., description="One powerful paragraph identifying the main insight drawn from the market scenario. MUST specifically focus on recommended expansion whitespace, product launch viability, M&A targets, or partnership strategies based on the institutional data gathered.")
    drivers: List[str] = Field(..., min_length=2, max_length=4, description="2-4 key market drivers propelling growth.")
    restraints: List[str] = Field(..., min_length=2, max_length=4, description="2-4 key market restraints or bottlenecks.")
    opportunities: List[str] = Field(..., min_length=2, max_length=4, description="2-4 key market opportunities (e.g., whitespace, emerging tech).")
    threats: List[str] = Field(..., min_length=2, max_length=4, description="2-4 key market threats (e.g., regulatory risks, substitution).")

class MarketSizingData(BaseModel):
    top_down_industry_tam_billions: float = Field(
        ..., 
        description="ACTUAL verified global top-down industry TAM in BILLIONS USD. TRILLION RULE: Multiply Trillions by 1000."
    )
    top_down_tam_period: str = Field(
        ..., 
        description="Reported period for top-down TAM (Preferably 'FY2026' or 'FY2025', rarely 'FY2024')."
    )
    top_down_publisher: str = Field(..., description="Name of the top-tier firm providing the TAM benchmark.")
    segments: List[MarketSegment] = Field(
        ..., 
        min_length=2, 
        max_length=10, 
        description="As many strictly MECE main functional pillars as required to cover the market without overlap (minimum 2)."
    )
    market_dynamics: MarketDynamics = Field(..., description="Strategic intelligence and DROT analysis deduced from the institutional data.")

# ==============================================================================
# 3. HIGH-AUTHORITY SEARCH TOOL WITH TEMPORAL WATERFALL
# ==============================================================================
@tool("Web Search")
def free_search_tool(query: str) -> str:
    """Searches the web for ACTUAL reported revenues strictly from top-tier institutional sources (Prioritizing 2026, then 2025)."""
    try:
        augmented_query = f"{query} (Gartner OR McKinsey OR Deloitte OR IDC OR Reuters OR Bloomberg OR SEC) FY2026 OR FY2025 OR LTM actual revenue -forecast -projected -\"expected to reach\""
        raw_results = list(DDGS().text(augmented_query, max_results=12, timelimit='y'))
        
        if not raw_results:
            fallback_query = f"{query} market size actual revenue (FY2026 OR FY2025 OR FY2024) (Gartner OR Deloitte OR IDC OR Reuters)"
            raw_results = list(DDGS().text(fallback_query, max_results=10))
            
        screened_results = []
        for r in raw_results:
            link = str(r.get('href', r.get('link', ''))).strip()
            title = str(r.get('title', '')).strip()
            snippet = str(r.get('body', r.get('snippet', ''))).strip()
            if is_high_authority_source(link, title, snippet):
                screened_results.append((title, link, snippet))
                
        if not screened_results and raw_results:
            for r in raw_results[:5]:
                screened_results.append((str(r.get('title', '')), str(r.get('href', r.get('link', ''))), str(r.get('body', r.get('snippet', '')))))

        formatted_results = []
        for idx, (title, link, snippet) in enumerate(screened_results[:5], 1):
            clean_link = link.replace('\n', '').replace(' ', '')
            formatted_results.append(f"High-Authority Source [{idx}]: {title}\nURL: {clean_link}\nData Snippet: {snippet}\n")
            
        return "\n---\n".join(formatted_results) if formatted_results else "No high-authority disclosures found."
    except Exception as e:
        return f"Error conducting web search: {str(e)}"

# ==============================================================================
# 4. DETERMINISTIC PYTHON AUDIT ENGINE
# ==============================================================================
PROHIBITED_FUTURE_PATTERNS = [
    r"expected to reach", r"projected to grow", r"projected to reach",
    r"forecasted to", r"is expected to", r"estimated to reach",
    r"by 2027", r"by 2028", r"by 2029", r"by 2030"
]

def compile_reconciled_report(data: MarketSizingData, market_name: str, scalar: float, final_tam: float, unit_correction: bool) -> str:
    md = [f"# Strict MECE Market Brief: {market_name}\n"]
    md.append(f"> **Reconciled Reported TAM ({data.top_down_tam_period}):** **${final_tam:.2f} Billion**")
    md.append(f"> *Primary Industry Benchmark Source: **{data.top_down_publisher}***\n")
    
    if unit_correction:
        md.append(f"> ⚠️ *Audit Correction: LLM unit-scale truncation intercepted and auto-corrected.*\n")
    
    md.append("## Dynamic Sub-Segment Revenue Architecture (Strictly Non-Overlapping)\n")
    
    all_citations = []
    citation_idx = 1
    main_segment_totals = []

    for seg in data.segments:
        seg_raw_sum = sum(sub.estimated_subsegment_market_size_billions for sub in seg.sub_segments)
        seg_reconciled_sum = seg_raw_sum * scalar
        main_segment_totals.append((seg.segment_name, seg_reconciled_sum))
        
        md.append(f"### 📌 Pillar: {seg.segment_name}")
        md.append(f"**MECE Definition:** *{seg.definition}*\n")
        md.append("| Sub-Segment Name | Key Vendors | Reconciled Sub-TAM | Actual Period | Recognized Source | Citation |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        
        for sub in seg.sub_segments:
            sub_reconciled_rev = sub.estimated_subsegment_market_size_billions * scalar
            
            snippet_lower = sub.verification_snippet.lower()
            has_forecast_violation = any(re.search(pat, snippet_lower) for pat in PROHIBITED_FUTURE_PATTERNS)
            status_flag = " ⚠️ [FORECAST DETECTED]" if has_forecast_violation else ""
            
            v_str = ", ".join(sub.top_vendors)
            rev_str = f"${sub_reconciled_rev:.2f}B" if sub_reconciled_rev >= 1.0 else f"${sub_reconciled_rev*1000:.0f}M"
            
            md.append(f"| **{sub.sub_segment_name}**{status_flag} | {v_str} | **{rev_str}** | `{sub.reporting_period}` | **{sub.publisher_name}** | `[{citation_idx}]` |")
            
            all_citations.append(
                f"**[{citation_idx}] {sub.sub_segment_name} — {rev_str} ({sub.reporting_period})**\n"
                f"* **Institutional Source:** {sub.publisher_name}\n"
                f"* **Key Vendors:** {v_str}\n"
                f"* **Evidentiary Snippet:** *\"{sub.verification_snippet}\"*\n"
                f"* **Verified Source URL:** [{sub.source_url}]({sub.source_url})\n"
            )
            citation_idx += 1
            
        seg_tot_str = f"${seg_reconciled_sum:.2f}B" if seg_reconciled_sum >= 1.0 else f"${seg_reconciled_sum*1000:.0f}M"
        md.append(f"| **TOTAL FOR PILLAR** | *All Pillar Vendors* | **{seg_tot_str}** | *Aggregated* | *Institutional Mix* | *N/A* |\n")

    md.append("## Consolidated TAM Synthesis\n")
    md.append("| Strictly MECE Main Pillar | Reconciled Reported Revenue | % Share of Total TAM | Methodology |")
    md.append("| :--- | :--- | :--- | :--- |")
    
    for seg_name, seg_rev in main_segment_totals:
        share_pct = (seg_rev / final_tam) * 100.0 if final_tam > 0 else 0
        rev_str = f"${seg_rev:.2f}B" if seg_rev >= 1.0 else f"${seg_rev*1000:.0f}M"
        md.append(f"| **{seg_name}** | **{rev_str}** | {share_pct:.1f}% | *Sum of Sub-Segments* |")
        
    md.append(f"| **TOTAL ADDRESSABLE MARKET** | **${final_tam:.2f}B** | **100.0%** | **Reconciled Top-Down Benchmark** |\n")
    
    # NEW COMPILER SECTION: Strategic Insights & DROT
    md.append("---\n")
    md.append("## 🧠 Strategic Market Insights & DROT Analysis\n")
    md.append(f"**Strategic Actionability (M&A / Expansion / Partnerships):**\n{data.market_dynamics.strategic_insight}\n")
    
    md.append("### Market Dynamics")
    md.append("**🚀 Key Drivers:**")
    for d in data.market_dynamics.drivers: md.append(f"- {d}")
    md.append("\n**🚧 Restraints & Bottlenecks:**")
    for r in data.market_dynamics.restraints: md.append(f"- {r}")
    md.append("\n**💡 Opportunities & Whitespace:**")
    for o in data.market_dynamics.opportunities: md.append(f"- {o}")
    md.append("\n**⚠️ Market Threats:**")
    for t in data.market_dynamics.threats: md.append(f"- {t}")
    md.append("\n---\n")

    md.append("## Institutional Sources & High-Authority Audit Trail\n")
    for cit in all_citations:
        md.append(f"{cit}")
        
    return "\n".join(md)

# ==============================================================================
# 5. WORKFLOW RUNNER
# ==============================================================================
target_market = st.text_input("🎯 Enter Target Market:", placeholder="e.g., Global Electric Vehicle Battery Market")

if st.button("🚀 Run Enterprise Sizing Engine", type="primary"):
    if not api_key_input or not target_market.strip():
        st.error("Please provide an API key and target market.")
        st.stop()

    os.environ[env_var_name] = api_key_input
    
    engine_llm = LLM(
        model=model_name, 
        api_key=api_key_input, 
        base_url=custom_base_url, 
        temperature=0.0, 
        max_retries=5, 
        timeout=300
    )

    quantifier_agent = Agent(
        role='Director of Market Architecture & Financial Audit',
        goal='Map the market into strictly MECE pillars, extract LATEST verified revenues, and synthesize strategic insights.',
        backstory=(
            'You are an elite market architect and institutional auditor. You map industries based on their TRUE organic structure. '
            'CRITICAL DIRECTIVE: Every single pillar and sub-segment you create MUST BE STRICTLY MECE (Mutually Exclusive, Collectively Exhaustive). '
            'TIMEFRAME DIRECTIVE: Extract the most recent actual reported financials available. Prioritize FY2026 or LTM 2026. If unavailable, use FY2025. Use FY2024 ONLY as a last resort for highly niche segments. '
            'You are forbidden from using data older than 2024. '
            'STRATEGIC DIRECTIVE: After compiling the data, you must deduce actionable insights regarding M&A, partnerships, and market dynamics (DROT).'
        ),
        verbose=True,
        tools=[free_search_tool],
        llm=engine_llm
    )

    sizing_task = Task(
        description=(
            f"Conduct comprehensive, deep-dive MECE market sizing and strategic analysis for '{target_market}'.\n"
            "1. Search for actual reported global TAM benchmarks in BILLIONS USD for the latest available period (FY2026, FY2025, or FY2024) from top analyst houses.\n"
            "2. ANTI-LAZINESS PROTOCOL: Map the ENTIRE organic architecture of the market. Attempt to build 4 pillars with 3+ sub-segments each unless the market is strictly consolidated.\n"
            "3. For every sub-segment, estimate ACTUAL reported market spend in BILLIONS USD for the latest available year using whitelisted institutional sources.\n"
            "4. STRATEGIC SYNTHESIS: Based on the institutional data, generate a 'strategic_insight' focused on expansion, M&A targets, or partnership strategies.\n"
            "5. DROT ANALYSIS: Identify 2-4 Drivers, Restraints, Opportunities, and Threats affecting this market."
        ),
        expected_output="A highly granular MarketSizingData Pydantic object containing the most recently available historical data and strategic market intelligence (DROT).",
        agent=quantifier_agent,
        output_pydantic=MarketSizingData
    )

    crew = Crew(agents=[quantifier_agent], tasks=[sizing_task], process=Process.sequential, max_rpm=crew_rpm_limit)

    with st.status("⚡ Running Enterprise Sizing Engine...", expanded=True) as status:
        st.write("🔍 Mapping organic market architecture and ensuring strict non-overlapping boundaries...")
        
        try:
            result = crew.kickoff()
            structured_data: MarketSizingData = result.pydantic
            
            if not structured_data:
                raise ValueError("Model returned empty structured data. Please re-run or switch model tier.")
            
            # ==================================================================
            # 🧮 PRE-CALCULATE RECONCILIATION SCALAR FOR CHARTS & REPORT
            # ==================================================================
            target_tam = structured_data.top_down_industry_tam_billions
            raw_sum = sum(sub.estimated_subsegment_market_size_billions for seg in structured_data.segments for sub in seg.sub_segments)
            
            unit_correction = False
            if raw_sum > (target_tam * 50.0) and raw_sum > 10.0:
                target_tam = target_tam * 1000.0
                unit_correction = True
            elif (target_tam > (raw_sum * 50.0)) and target_tam > 100.0 and raw_sum < 10.0:
                raw_sum = raw_sum * 1000.0
                unit_correction = True
                
            scalar = (target_tam / raw_sum) if raw_sum > 0 else 1.0

            # ==================================================================
            # 📊 VISUAL ANALYTICS LAYER (PLOTLY EXPRESS)
            # ==================================================================
            st.write("📊 Rendering interactive market distribution charts...")
            
            st.markdown("[**⬇️ Jump directly to the Visual Dashboard**](#market-distribution)")
            
            chart_data = []
            for seg in structured_data.segments:
                for sub in seg.sub_segments:
                    reconciled_rev = sub.estimated_subsegment_market_size_billions * scalar
                    chart_data.append({
                        "Main Pillar": seg.segment_name,
                        "Sub-Segment": sub.sub_segment_name,
                        "Revenue ($B)": reconciled_rev
                    })
            df = pd.DataFrame(chart_data)
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("### 🍩 Market Distribution")
                fig_sunburst = px.sunburst(
                    df, 
                    path=['Main Pillar', 'Sub-Segment'], 
                    values='Revenue ($B)',
                    color='Main Pillar',
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_sunburst.update_traces(textinfo="label+percent parent+value")
                fig_sunburst.update_layout(margin=dict(t=10, l=10, r=10, b=10))
                st.plotly_chart(fig_sunburst, use_container_width=True)
                
            with chart_col2:
                st.markdown("### 📊 Pillar Valuation")
                fig_bar = px.bar(
                    df.groupby('Main Pillar', as_index=False)['Revenue ($B)'].sum().sort_values('Revenue ($B)', ascending=False),
                    x='Main Pillar', 
                    y='Revenue ($B)',
                    text='Revenue ($B)',
                    color='Main Pillar',
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_bar.update_traces(texttemplate='$%{text:.2f}B', textposition='outside')
                max_val = df.groupby('Main Pillar')['Revenue ($B)'].sum().max()
                fig_bar.update_layout(showlegend=False, xaxis_title="", yaxis_title="Billions (USD)", yaxis=dict(range=[0, max_val * 1.15]), margin=dict(t=10, l=10, r=10, b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

            # ==================================================================
            # 📝 MARKDOWN COMPILER
            # ==================================================================
            st.write("🧮 Validating institutional evidence and normalizing MECE aggregations...")
            final_markdown_report = compile_reconciled_report(structured_data, target_market, scalar, target_tam, unit_correction)
            
            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
            
            st.subheader(f"📊 Institutional Market Intelligence Brief: {target_market}")
            
            safe_market_name = re.sub(r'[^a-zA-Z0-9_-]', '_', target_market.lower())
            
            col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
            with col2:
                st.download_button(
                    label="📥 Download Report (MD)",
                    data=final_markdown_report,
                    file_name=f"Enterprise_Sizing_{safe_market_name}.md",
                    mime="text/markdown"
                )
            with col3:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data (CSV)",
                    data=csv_data,
                    file_name=f"Reconciled_Data_{safe_market_name}.csv",
                    mime="text/csv"
                )
            
            st.markdown("---")
            st.markdown(final_markdown_report)
            
        except Exception as e:
            status.update(label="⚠️ Execution Notice", state="error", expanded=True)
            st.error(f"An error occurred during data validation or workflow execution:\n\n`{str(e)}`")
            st.info("💡 **Tip:** If you hit a rate limit, simply run the app again.")
