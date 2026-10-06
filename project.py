import streamlit as st
import pandas as pd
import plotly.express as px

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Study Planner",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Smart Study Planner")
st.write("AI-Based Study Planner with RAG Chatbot")


# --------------------------------------------------
# SIDEBAR - STUDY PLANNER
# --------------------------------------------------

st.sidebar.header("📅 Study Planner")

name = st.sidebar.text_input("Student Name")

subjects = st.sidebar.text_input(
    "Subjects",
    "Java, DBMS, Python, Operating Systems"
)

study_hours = st.sidebar.number_input(
    "Study Hours Per Day",
    min_value=1,
    max_value=12,
    value=4
)

difficulty = st.sidebar.selectbox(
    "Difficulty",
    ["Easy", "Medium", "Hard"]
)

exam_days = st.sidebar.number_input(
    "Days Remaining",
    min_value=1,
    max_value=100,
    value=15
)


# --------------------------------------------------
# STUDY PLAN
# --------------------------------------------------

if st.sidebar.button("Generate Study Plan"):

    subject_list = [
        x.strip()
        for x in subjects.split(",")
        if x.strip()
    ]

    hours_each = study_hours / len(subject_list)

    data = []

    for subject in subject_list:

        if difficulty == "Hard":
            priority = "High"

        elif difficulty == "Medium":
            priority = "Medium"

        else:
            priority = "Low"

        data.append({
            "Subject": subject,
            "Study Hours": round(hours_each, 2),
            "Priority": priority
        })

    df = pd.DataFrame(data)

    st.success(
        f"Study plan created for {name if name else 'Student'}!"
    )

    st.header("📅 Your Study Plan")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.header("📊 Study Hours")

    fig = px.bar(
        df,
        x="Subject",
        y="Study Hours",
        title="Study Hours Per Subject"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# --------------------------------------------------
# AI RECOMMENDATION
# --------------------------------------------------

st.header("🤖 AI Study Recommendation")

if difficulty == "Hard":

    recommendation = (
        "Focus more on difficult subjects. "
        "Practice important concepts and previous questions."
    )

elif difficulty == "Medium":

    recommendation = (
        "Maintain a balanced study schedule. "
        "Divide your time between learning, practice and revision."
    )

else:

    recommendation = (
        "Focus mainly on revision and practice questions."
    )


if exam_days <= 7:

    recommendation += (
        " Your exam is very close. "
        "Prioritize important topics and previous papers."
    )

elif exam_days <= 15:

    recommendation += (
        " Complete your syllabus soon "
        "and keep enough time for revision."
    )

else:

    recommendation += (
        " You have enough time. "
        "Follow a consistent daily schedule."
    )


st.info(recommendation)


# --------------------------------------------------
# RAG CHATBOT
# --------------------------------------------------

st.header("🤖 Study Material Chatbot")
st.write(
    "Ask questions from your uploaded study material."
)


# Create RAG only when button is clicked

if st.button("Load Study Material"):

    try:

        # Load PDF
        loader = PyPDFLoader("study_material.pdf")

        documents = loader.load()

        # Split PDF into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(documents)

        # Embedding model
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        # Create vector database
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./study_db"
        )

        st.session_state.vector_db = vector_db

        st.success(
            "Study material loaded successfully! ✅"
        )

    except Exception as e:

        st.error(
            "Error loading PDF. Make sure "
            "study_material.pdf is present and Ollama is running."
        )

        st.error(str(e))


# --------------------------------------------------
# CHATBOT
# --------------------------------------------------

question = st.text_input(
    "Ask a question from your study material:"
)


if st.button("Ask AI"):

    if "vector_db" not in st.session_state:

        st.warning(
            "First click 'Load Study Material'."
        )

    elif question.strip() == "":

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            # Retrieve relevant documents
            docs = st.session_state.vector_db.similarity_search(
                question,
                k=4
            )

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            # Ollama LLM
            llm = ChatOllama(
                model="llama3.2",
                temperature=0
            )

            prompt = f"""
You are a helpful study assistant.

Answer the student's question using ONLY
the study material given below.

If the answer is not available in the
study material, say:

"I could not find this information
in the uploaded study material."

Study Material:
{context}

Student Question:
{question}
"""

            response = llm.invoke(prompt)

            st.subheader("💬 AI Answer")

            st.write(response.content)

        except Exception as e:

            st.error(
                "Chatbot error. Check whether Ollama "
                "and the required models are installed."
            )

            st.error(str(e))


# --------------------------------------------------
# STUDY TIPS
# --------------------------------------------------

st.header("💡 Study Tips")

st.write("✅ Study difficult subjects first.")
st.write("✅ Take short breaks during study.")
st.write("✅ Revise regularly.")
st.write("✅ Practice previous question papers.")
st.write("✅ Ask the chatbot questions from your PDF.")

