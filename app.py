import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg

import os
import re
import io
import streamlit as st
import plotly.express as px
import pandas as pd
from pydantic import BaseModel, Field
from typing import List
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS
from markdown_pdf import MarkdownPdf, Section

# ==============================================================================
# 1. UI SETUP & STATE MANAGEMENT
# ==============================================================================
st.set_page_config(
    page_title="Institutional Market Sizing Engine",
    page_icon="📈",
    layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sizing_data" not in st.session_state:
    st.session_state.sizing_data = None
if "strategy_complete" not in st.session_state:
    st.session_state.strategy_complete = False
if "drot_markdown" not in st.session_state:
    st.session_state.drot_markdown = ""
if "api_key_cache" not in st.session_state:
    st.session_state.api_key_cache = ""

st.title("📈 Institutional Market Sizing Engine")
st.markdown("This engine utilizes a **Two-Phase Architecture**: 1. Quantitative MECE TAM Extraction. 2. Intent-Driven Strategic DROT Synthesis.")

with st.sidebar:
    st.header("🔑 API Configuration")
    provider = st.radio("Select Free API Provider:", ["Google Gemini (Free Tier)", "Groq Cloud (100% Free Backup)"])
    
    if provider == "Google Gemini (Free Tier)":
        api_key_input = st.text_input("Gemini API Key", type="password")
        model_name = "gemini/gemini-1.5-flash"  # <-- FIXED MODEL VERSION
        env_var_name = "GEMINI_API_KEY"
        custom_base_url = None
    else:
        api_key_input = st.text_input("Groq API Key", type="password")
        model_name = "openai/mixtral-8x7b-32768"
        env_var_name = "GROQ_API_KEY"
        custom_base_url = "https://api.groq.com/openai/v1"

    st.session_state.api_key_cache = api_key_input

# ==============================================================================
# 2. PYDANTIC SCHEMAS (CONSTRAINTS MOVED TO TEXT FOR GOOGLE COMPATIBILITY)
# ==============================================================================
HIGH_AUTHORITY_PATTERNS = [
    "mckinsey", "bcg.com", "bain.com", "deloitte", "pwc.com", "ey.com", "kpmg", "accenture",
    "gartner", "idc.com", "forrester", "statista", "bloomberg", "reuters", "pitchbook", 
    "sec.gov", "investor.", "wsj.com", "ft.com", "cnbc.com", "forbes.com"
]

def is_high_authority_source(url: str, title: str, snippet: str) -> bool:
    combined_text = f"{url.lower()} {title.lower()} {snippet.lower()}"
    return any(pattern in combined_text for pattern in HIGH_AUTHORITY_PATTERNS)

class SubSegmentData(BaseModel):
    sub_segment_name: str = Field(..., description="Name of the specific functional sub-segment")
    top_vendors: List[str] = Field(..., description="Top key market players operating in this sub-segment")
    estimated_subsegment_market_size_billions: float = Field(..., description="Actual reported market valuation in BILLIONS USD.")
    reporting_period: str = Field(..., description="Latest available period: 'FY2026', 'LTM', 'FY2025'")
    publisher_name: str = Field(..., description="Recognized institutional source.")
    source_url: str = Field(..., description="Hyperlink URL from a high-authority domain.")
    verification_snippet: str = Field(..., description="Verbatim textual quote proving reported figures.")

class MarketSegment(BaseModel):
    segment_name: str = Field(..., description="Strictly MECE main functional segment name")
    definition: str = Field(..., description="Boundary definition proving why this pillar never overlaps with others")
    sub_segments: List[SubSegmentData] = Field(..., description="Provide at least 2 distinct, strictly non-overlapping sub-segments.")

class MarketSizingData(BaseModel):
    top_down_industry_tam_billions: float = Field(...)
    top_down_tam_period: str = Field(...)
    top_down_publisher: str = Field(...)
    segments: List[MarketSegment] = Field(..., description="Provide at least 2 strictly MECE main functional pillars.")

class MarketDynamics(BaseModel):
    strategic_insight: str = Field(..., description="One powerful paragraph identifying the main insight drawn based on the user's chosen strategic intent.")
    drivers: List[str] = Field(..., description="List 2 to 4 key market drivers.")
    restraints: List[str] = Field(..., description="List 2 to 4 key market restraints.")
    opportunities: List[str] = Field(..., description="List 2 to 4 key market opportunities.")
    threats: List[str] = Field(..., description="List 2 to 4 key market threats.")

# ==============================================================================
# 3. SEARCH TOOL 
# ==============================================================================
@tool("Web Search")
def free_search_tool(query: str) -> str:
    """Searches the web for ACTUAL reported revenues strictly from top-tier institutional sources."""
    try:
        augmented_query = f"{query} (Gartner OR McKinsey OR Deloitte OR IDC OR Reuters OR Bloomberg OR SEC) FY2026 OR FY2025 OR LTM actual revenue -forecast -projected"
        raw_results = list(DDGS().text(augmented_query, max_results=12, timelimit='y'))
        
        screened_results = []
        for r in raw_results:
            link, title, snippet = str(r.get('href', '')), str(r.get('title', '')), str(r.get('body', ''))
            if is_high_authority_source(link, title, snippet):
                screened_results.append((title, link, snippet))
                
        if not screened_results and raw_results:
            for r in raw_results[:5]: screened_results.append((str(r.get('title', '')), str(r.get('href', '')), str(r.get('body', ''))))

        formatted_results = [f"Source [{idx}]: {t}\nURL: {l}\nData: {s}\n" for idx, (t, l, s) in enumerate(screened_results[:5], 1)]
        return "\n---\n".join(formatted_results) if formatted_results else "No high-authority disclosures found."
    except Exception as e: return f"Error: {str(e)}"

# ==============================================================================
# 4. MARKDOWN COMPILERS 
# ==============================================================================
def compile_sizing_report(data: MarketSizingData, market_name: str, scalar: float, final_tam: float, unit_correction: bool):
    md = [f"# Strict MECE Market Brief: {market_name}\n"]
    md.append(f"> **Reconciled Reported TAM ({data.top_down_tam_period}):** **${final_tam:.2f} Billion**")
    md.append(f"> *Primary Industry Benchmark Source: **{data.top_down_publisher}***\n")
    if unit_correction: md.append(f"> ⚠️ *Audit Correction: LLM unit-scale truncation auto-corrected.*\n")
    md.append("## Dynamic Sub-Segment Revenue Architecture\n")
    
    all_citations, citation_idx, main_segment_totals = [], 1, []

    for seg in data.segments:
        seg_raw_sum = sum(sub.estimated_subsegment_market_size_billions for sub in seg.sub_segments)
        seg_reconciled_sum = seg_raw_sum * scalar
        main_segment_totals.append((seg.segment_name, seg_reconciled_sum))
        
        md.extend([f"### 📌 Pillar: {seg.segment_name}", f"**MECE Definition:** *{seg.definition}*\n"])
        md.extend(["| Sub-Segment | Key Vendors | Reconciled Sub-TAM | Period | Source | Cit. |", "| :--- | :--- | :--- | :--- | :--- | :--- |"])
        
        for sub in seg.sub_segments:
            sub_rev = sub.estimated_subsegment_market_size_billions * scalar
            rev_str = f"${sub_rev:.2f}B" if sub_rev >= 1.0 else f"${sub_rev*1000:.0f}M"
            v_str = ", ".join(sub.top_vendors)
            md.append(f"| **{sub.sub_segment_name}** | {v_str} | **{rev_str}** | `{sub.reporting_period}` | **{sub.publisher_name}** | `[{citation_idx}]` |")
            all_citations.append(f"**[{citation_idx}] {sub.sub_segment_name} — {rev_str}**\n* **Vendors:** {v_str}\n* **Snippet:** *\"{sub.verification_snippet}\"*\n* **URL:** [{sub.source_url}]({sub.source_url})\n")
            citation_idx += 1
            
        seg_tot = f"${seg_reconciled_sum:.2f}B" if seg_reconciled_sum >= 1.0 else f"${seg_reconciled_sum*1000:.0f}M"
        md.append(f"| **TOTAL FOR PILLAR** | *All Pillar Vendors* | **{seg_tot}** | *Aggregated* | *N/A* | *N/A* |\n")

    md.extend(["## Consolidated TAM Synthesis\n", "| Strictly MECE Main Pillar | Reconciled Revenue | % Share of TAM |", "| :--- | :--- | :--- |"])
    for seg_name, seg_rev in main_segment_totals:
        md.append(f"| **{seg_name}** | **${seg_rev:.2f}B** | {(seg_rev/final_tam)*100:.1f}% |")
    md.append(f"| **TOTAL TAM** | **${final_tam:.2f}B** | **100.0%** |\n---\n")
    
    citations_md = "\n## Institutional Audit Trail\n" + "\n".join(all_citations)
    return "\n".join(md), citations_md

def compile_drot_report(drot: MarketDynamics, intent: str) -> str:
    md = [f"## 🧠 Strategic Market Insights & DROT Analysis (Target: {intent})\n"]
    md.extend([f"**Strategic Actionability:**\n{drot.strategic_insight}\n", "### Market Dynamics"])
    md.append("**🚀 Key Drivers:**"); md.extend([f"- {d}" for d in drot.drivers])
    md.append("\n**🚧 Restraints & Bottlenecks:**"); md.extend([f"- {r}" for r in drot.restraints])
    md.append("\n**💡 Opportunities & Whitespace:**"); md.extend([f"- {o}" for o in drot.opportunities])
    md.append("\n**⚠️ Market Threats:**"); md.extend([f"- {t}" for t in drot.threats])
    return "\n".join(md) + "\n\n---\n"

def export_to_pdf(markdown_text: str) -> bytes:
    pdf = MarkdownPdf(toc_level=2)
    pdf.add_section(Section(markdown_text, toc=False))
    out = io.BytesIO()
    pdf.save_bytes(out)
    return out.getvalue()

# ==============================================================================
# 5. PHASE 1: TAM & SEGMENTATION LOGIC
# ==============================================================================
target_market = st.text_input("🎯 Enter Target Market:", placeholder="e.g., Global Electric Vehicle Battery Market")

if st.button("🚀 Phase 1: Extract Market Architecture (TAM)", type="primary"):
    if not api_key_input or not target_market.strip(): st.error("Provide API key and market."); st.stop()
    os.environ[env_var_name] = api_key_input
    
    st.session_state.sizing_data = None
    st.session_state.strategy_complete = False
    st.session_state.chat_history = []
    
    llm_kwargs = {"model": model_name, "api_key": api_key_input, "temperature": 0.0}
    if custom_base_url: llm_kwargs["base_url"] = custom_base_url
    engine_llm = LLM(**llm_kwargs)
    
    quantifier = Agent(role='Market Architect', goal='Map MECE pillars and extract revenues.', backstory='Elite auditor.', tools=[free_search_tool], llm=engine_llm)
    sizing_task = Task(description=f"Map MECE TAM for '{target_market}'.", expected_output="MarketSizingData JSON", agent=quantifier, output_pydantic=MarketSizingData)
    
    with st.status("⚡ Extracting Market Architecture...", expanded=True) as status:
        res = Crew(agents=[quantifier], tasks=[sizing_task], process=Process.sequential).kickoff()
        struct_data: MarketSizingData = res.pydantic
        
        target_tam = struct_data.top_down_industry_tam_billions
        raw_sum = sum(sub.estimated_subsegment_market_size_billions for seg in struct_data.segments for sub in seg.sub_segments)
        uc = False
        if raw_sum > (target_tam * 50) and raw_sum > 10: target_tam *= 1000; uc = True
        elif target_tam > (raw_sum * 50) and target_tam > 100 and raw_sum < 10: raw_sum *= 1000; uc = True
        scalar = (target_tam / raw_sum) if raw_sum > 0 else 1.0

        chart_data = [{"Main Pillar": seg.segment_name, "Sub-Segment": sub.sub_segment_name, "Revenue ($B)": sub.estimated_subsegment_market_size_billions * scalar} for seg in struct_data.segments for sub in seg.sub_segments]
        
        sizing_md, citations_md = compile_sizing_report(struct_data, target_market, scalar, target_tam, uc)
        
        st.session_state.sizing_data = {
            "market": target_market, "df": pd.DataFrame(chart_data), 
            "sizing_md": sizing_md, "citations_md": citations_md, "raw_json": struct_data.model_dump_json()
        }
        status.update(label="✅ Phase 1 Complete!", state="complete")
        st.rerun()

# ==============================================================================
# 6. DASHBOARD & PHASE 2: HUMAN-IN-THE-LOOP PARAMETER SELECTION
# ==============================================================================
if st.session_state.sizing_data:
    sd = st.session_state.sizing_data
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🍩 Market Distribution")
        fig1 = px.sunburst(sd["df"], path=['Main Pillar', 'Sub-Segment'], values='Revenue ($B)', color='Main Pillar', color_discrete_sequence=px.colors.qualitative.Prism)
        fig1.update_traces(textinfo="label+percent parent+value"); st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.markdown("### 📊 Pillar Valuation")
        fig2 = px.bar(sd["df"].groupby('Main Pillar', as_index=False)['Revenue ($B)'].sum().sort_values('Revenue ($B)', ascending=False), x='Main Pillar', y='Revenue ($B)', text='Revenue ($B)', color='Main Pillar', color_discrete_sequence=px.colors.qualitative.Prism)
        fig2.update_traces(texttemplate='$%{text:.2f}B', textposition='outside'); st.plotly_chart(fig2, use_container_width=True)

    if not st.session_state.strategy_complete:
        st.markdown("---")
        st.markdown("## 🎯 Phase 2: Define Strategic Intent (Human-in-the-Loop)")
        st.info("The quantitative architecture is complete. How do you intend to action this market data?")
        
        strategic_intent = st.radio(
            "Select your primary goal to generate a customized DROT analysis:", 
            ["Market Expansion", "New Product Launch", "Mergers & Acquisitions (M&A)", "Strategic Partnerships"]
        )
        
        if st.button("⚡ Generate Custom Strategy & DROT", type="primary"):
            os.environ[env_var_name] = st.session_state.api_key_cache
            
            llm_kwargs = {"model": model_name, "api_key": st.session_state.api_key_cache, "temperature": 0.3}
            if custom_base_url: llm_kwargs["base_url"] = custom_base_url
            strat_llm = LLM(**llm_kwargs)
            
            strat_agent = Agent(role='Chief Strategy Officer', goal='Generate actionable DROT based on intent.', backstory='Elite strategist.', llm=strat_llm)
            strat_task = Task(description=f"Analyze this market JSON: {sd['raw_json']}. The client's intent is: **{strategic_intent}**. Generate a highly specific DROT tailored ONLY to this intent.", expected_output="MarketDynamics JSON", agent=strat_agent, output_pydantic=MarketDynamics)
            
            with st.spinner(f"Synthesizing {strategic_intent} Strategy..."):
                strat_res = Crew(agents=[strat_agent], tasks=[strat_task], process=Process.sequential).kickoff()
                st.session_state.drot_markdown = compile_drot_report(strat_res.pydantic, strategic_intent)
                st.session_state.strategy_complete = True
                st.rerun()

# ==============================================================================
# 7. FINAL DOSSIER, EXPORT & RAG CHATBOT 
# ==============================================================================
if st.session_state.sizing_data and st.session_state.strategy_complete:
    sd = st.session_state.sizing_data
    full_dossier = f"{sd['sizing_md']}\n{st.session_state.drot_markdown}\n{sd['citations_md']}"
    
    st.markdown("---")
    st.markdown(full_dossier)
    st.markdown("---")
    
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', sd["market"].lower())
    b1, b2, b3 = st.columns(3)
    with b1: st.download_button("📥 Download (MD)", full_dossier, f"Dossier_{safe_name}.md", "text/markdown", use_container_width=True)
    with b2: st.download_button("📄 Download (PDF)", export_to_pdf(full_dossier), f"Dossier_{safe_name}.pdf", "application/pdf", use_container_width=True)
    with b3: st.download_button("📊 Download (CSV)", sd["df"].to_csv(index=False).encode('utf-8'), f"Data_{safe_name}.csv", "text/csv", use_container_width=True)

    st.markdown("---")
    st.subheader("💬 Ask the Strategy Assistant")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if user_query := st.chat_input("Ask a follow-up about vendors, gaps, or strategy..."):
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"): st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing dossier..."):
                try:
                    import litellm
                    os.environ[env_var_name] = st.session_state.api_key_cache
                    sys_prompt = f"You are a strategic assistant. Use ONLY this report to answer:\n---\n{full_dossier}"
                    response = litellm.completion(model=model_name, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_query}], api_key=st.session_state.api_key_cache, base_url=custom_base_url)
                    bot_reply = response.choices[0].message.content
                    st.markdown(bot_reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                    st.rerun()
                except Exception as e: st.error(f"Chatbot Error: {str(e)}")
