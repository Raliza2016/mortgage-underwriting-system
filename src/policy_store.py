import re
import os

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

FALLBACK_POLICIES = """
1.1 Credit Score Requirements
Minimum credit score for conventional loans: 620
Preferred credit score: 700+
Excellent tier (740+): Best rates available, no additional requirements.
Very Good tier (700-739): Favorable rates, standard documentation.
Good tier (660-699): Standard rates, may require compensating factors for high DTI.
Fair tier (620-659): Higher rates, compensating factors required.
Below 620: Does not meet conventional loan requirements.

1.2 Derogatory Credit Items
Bankruptcies: 4-year waiting period post-discharge (Chapter 7); 2 years for Chapter 13.
Foreclosures: 7-year waiting period from completion date.
Short Sales: 4-year waiting period.
Late Payments: No 30-day lates in the last 12 months preferred.
Collections: Must be paid or resolved prior to closing (exceptions for medical).

2.1 Income Verification
W-2 Employees: Last 2 years W-2s and 30-day pay stubs required.
Self-Employed: Last 2 years tax returns (personal and business), year-to-date P&L.
Income must be stable and likely to continue for at least 3 years.
Rental income: 75% of gross rental income may be used.

2.2 Debt-to-Income Ratio
Front-end ratio (housing): Maximum 28-31% of gross monthly income.
Back-end ratio (total debt): Maximum 43-50% depending on compensating factors.
Compensating factors: Significant reserves, excellent credit, low LTV.

2.3 Self-Employment Income
Two years of self-employment required.
Income averaged over 2 years.
Year-over-year income decline requires explanation letter.

3.1 Down Payment and Reserves
Minimum down payment: 3% (conventional), 3.5% (FHA), 0% (VA/USDA).
Reserves: Minimum 2 months PITI after closing.
Gift funds: Acceptable with gift letter; no repayment obligation.

3.2 Large Deposits
Any deposit exceeding 25% of monthly qualifying income requires sourcing.
Documentation: Bank statements, transfer records, or gift letter.

4.1 Property and Collateral
LTV: Maximum 97% conventional, 96.5% FHA.
Appraisal: Full appraisal required on all transactions.
Property condition: Must meet minimum habitability standards.
Non-warrantable condos and unique property types require additional review.

4.2 LTV Guidelines
80% or below: No PMI required, best rates.
80.01-90%: PMI required.
90.01-97%: PMI required, limited programs.
Above 97%: Not eligible for conventional financing.
"""


def create_policy_store(policy_pdf_path: str = "underwriting_policies.pdf"):
    """
    Create a vector store from underwriting policies.
    Falls back to built-in policies if PDF is not available.
    """
    if PDF_SUPPORT and os.path.exists(policy_pdf_path):
        loader = PyPDFLoader(policy_pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        policy_chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings(
            base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        )
        vectorstore = Chroma.from_documents(
            documents=policy_chunks,
            embedding=embeddings,
            collection_name="underwriting_policies",
        )
        return vectorstore

    return None


def retrieve_relevant_policies(query: str, vectorstore=None) -> str:
    """
    Retrieve relevant policy sections for a given query.
    Uses vector search if available, otherwise returns full policy text.
    """
    if vectorstore is not None:
        docs = vectorstore.similarity_search(query, k=6)
        section_map = {}
        for doc in docs:
            text = doc.page_content.strip()
            match = re.match(r"^\d+\.\d+\s+[A-Za-z ].+", text)
            section = match.group(0) if match else "OTHER"
            if section not in section_map:
                section_map[section] = text
            else:
                if text not in section_map[section]:
                    section_map[section] += "\n" + text
        return "\n\n".join(section_map.values())

    query_lower = query.lower()
    relevant = []
    sections = FALLBACK_POLICIES.strip().split("\n\n")
    for section in sections:
        keywords = query_lower.split()
        if any(kw in section.lower() for kw in keywords):
            relevant.append(section)
    return "\n\n".join(relevant) if relevant else FALLBACK_POLICIES
