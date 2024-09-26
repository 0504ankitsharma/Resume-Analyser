import streamlit as st
import fitz
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
# Initialize the Google Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def extract_text_from_pdf(file):
    """Extract text from PDF."""
    text = ""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def get_gemini_response(prompt):
    """Function to load Google Gemini model and provide queries as response."""
    response = model.generate_content([prompt])
    return response.text

def analyze_resume(text):
    prompt = (
        "You are an expert resume analyst. Analyze the following resume text and provide the following:\n\n"
        "1. A brief summary of the resume, including the candidate's main interests and the fields they are most passionate about.\n"
        "2. A detailed percentage distribution of fields/domains present in the resume, with keywords extracted from the resume. Ensure the total sums up to 100%.\n"
        "3. A overall explaination for each domain in brief according to the resume."
        "Here is an example of how the output should look like:\n\n"
        "### Summary\n"
        "The resume indicates a strong background and interest in machine learning, data science, and software development. The candidate has worked on several projects involving machine learning algorithms, data preprocessing, and building software applications. They have demonstrated proficiency in Python, Java, and various machine learning frameworks. The candidate is passionate about solving complex problems using AI and has a keen interest in continuing to develop their skills in this area.\n\n"
        "### Percentage Distribution of Fields/Domains\n"
        "Note: only include standard technologies as keywords.\n"
        "- *Machine Learning (ML)*: 40%\n"
        "  - Keywords: Algorithms, Keras, PyTorch, Scikit-Learn, Predictive Models\n"
        "- *Data Science (DS)*: 30%\n"
        "  - Keywords: Data Analysis, Pandas, NumPy, Visualization, Statistical Methods\n"
        "- *Software Development (SD)*: 30%\n"
        "  - Keywords: Python, Java, Software Engineering, APIs, Git\n"
    )
    analysis = get_gemini_response(prompt + text)
    return analysis

def extract_domains_from_analysis(analysis):
    """Extracts domain names from the analysis text."""
    domain_prompt = (
        "Based on the following resume analysis, extract and return the standard domain/field/keywords names present in the resume. "
        "Provide only the domain names separated by commas:\n\n"
        f"{analysis}"
    )
    domain_response = get_gemini_response(domain_prompt)
    return domain_response.strip()

def generate_mcq_questions(analysis, selected_domain):
    num_questions = 20  # Default to 20 questions
    prompt_template = (
        f"*Subject:* {selected_domain}\n\n"
        f"*Bloom Taxonomy Levels:*\n\n"
        f"- *Analysis:* Identify the relationships between concepts, analyze data, identify patterns and causes, and draw conclusions.\n"
        f"- *Apply:* Use learned concepts to solve problems, complete tasks, and apply principles to new situations.\n"
        f"- *Evaluate:* Assess the value or quality of something, make judgments, and justify decisions.\n\n"
        f"*MCQ Prompt:*\n\n"
        f"Generate {num_questions} code snippet types of multiple-choice questions (MCQs) for the subject of {selected_domain} aligned with the following Bloom Taxonomy levels:\n\n"
        f"- 8 questions at the *Analysis* level:\n"
        f"- 6 questions at the *Apply* level:\n"
        f"- 6 questions at the *Evaluate* level:\n\n"
        f"*Format:*\n\n"
        f"- Each question should have 4 options (A, B, C, D)\n"
        f"- Each question should have a clear and concise stem\n"
        f"- Each option should be plausible, but only one should be correct\n\n"
        f"Example:\n\n"
        f"1. What is the time complexity of sorting an array using the merge sort algorithm? (Multiple-choice)\n"
        f"a. O(n)\n"
        f"b. O(log n)\n"
        f"c. O(n log n)\n"
        f"d. O(n^2)\n\n"
        f"Answers:\n\n"
        f"1. c\n\n"
        f"Note: Ensure there are exactly {num_questions} questions in total with a random mix of question types."
    )
    questions = get_gemini_response(prompt_template)
    return questions

def generate_coding_questions(analysis, selected_domain):
    num_questions = 20  # Default to 20 questions
    prompt_template = (
        f"You are an expert in the {selected_domain} field. Based on the analysis of the candidate's resume, which shows {analysis}, "
        f"generate {num_questions} coding challenges that reflect fundamental to medium-level real-world scenarios and problems encountered in {selected_domain}. "
        f"The coding challenges should:\n\n"
        f"1. Be relevant to what a recruiter might ask a final-year student.\n"
        f"2. Focus on practical coding tasks.\n"
        f"3. Include identifying errors, completing code snippets, and writing simple to moderate algorithms.\n"
        f"4. Cover a range of difficulty levels from fundamental concepts to medium topics.\n"
        f"5. Be clear, concise, and directly related to the skills highlighted in the resume.\n"
        f"6. Include examples of typical coding questions.\n\n"
        f"Example:\n\n"
        f"1. Write a function to reverse a string. (Coding challenge)\n"
        f"```python\n"
        f"def reverse_string(s):\n"
        f"    return s[::-1]\n"
        f"```\n\n"
        f"2. Identify the error in the following code snippet and correct it. (Open-ended)\n"
        f"```python\n"
        f"def sum_of_squares(n):\n"
        f"    total = 0\n"
        f"    for i in range(n):\n"
        f"        total += i**2\n"
        f"    return total\n"
        f"```\n"
        f"Error: The range should be `range(n+1)` to include `n`.\n\n"
        f"3. Complete the following function to check if a number is prime. (Fill-in-the-blank)\n"
        f"```python\n"
        f"def is_prime(n):\n"
        f"    if n <= 1:\n"
        f"        return False\n"
        f"    for i in range(2, n):\n"
        f"        if n % i == 0:\n"
        f"            return False\n"
        f"    return True\n"
        f"```\n\n"
        f"Ensure there are exactly {num_questions} coding questions with a mix of the above types."
    )
    questions = get_gemini_response(prompt_template)
    return questions

def generate_interview_questions(analysis, selected_domain):
    num_questions = 20  # Default to 20 questions
    prompt_template = (
        f"Based on the candidate's resume {analysis} and the identified skills, experience, and education, generate a set of {num_questions} interview questions "
        f"that assess their fit for the position at our company. The questions should cover topics such as problem-solving abilities, leadership skills, "
        f"communication skills, cultural fit, etc. Additionally, include follow-up questions to probe deeper into the candidate's responses and evaluate their thought process. "
        f"Note:The questions should be specific to the field of {selected_domain}."
        f"Questions should be both knowledge base and industry application level also which include practical application of knowledge but only based on {selected_domain}."
        f"You need to take reference from the resume analysis but the questions generated should be strictly based on {selected_domain}."
    )
    questions = get_gemini_response(prompt_template)
    return questions

def split_questions_answers(quiz_response):
    """Function that splits the questions and answers from the quiz response."""
    if "Answers:" in quiz_response:
        questions = quiz_response.split("Answers:")[0]
        answers = quiz_response.split("Answers:")[1]
    else:
        questions = quiz_response
        answers = "Answers section not found in the response."
    return questions, answers

def main():
    st.title("Resume Analyzer")
    st.subheader("Your personal expert for placement")

    if'selected_domain' not in st.session_state:
        st.session_state.selected_domain = "DSA"

    if 'analysis' not in st.session_state:
        st.session_state.analysis = None

    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    uploaded_file = st.file_uploader("Upload your resume PDF", type="pdf")
    if uploaded_file is not None and uploaded_file!= st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        resume_text = extract_text_from_pdf(uploaded_file)
        
        with st.spinner("Analyzing Resume..."):
            st.session_state.analysis = analyze_resume(resume_text)
        
        st.write("Analysis Result:")
        st.write(st.session_state.analysis)
        
        # Extract domains from the analysis
        domain_response = extract_domains_from_analysis(st.session_state.analysis)
        domains = [domain.strip() for domain in domain_response.split(',')]
        default_domains = ["DSA", "DBMS", "Programming Basics"] + domains

        def update_selected_domain():
            st.session_state.selected_domain = st.session_state.domain_select

        st.selectbox(
            "Select Domain for Questions:",
            default_domains,
            key="domain_select",
            index=default_domains.index(st.session_state.selected_domain) if st.session_state.selected_domain in default_domains else 0,
            on_change=update_selected_domain
        )
        
        question_type = st.selectbox("Select Question Type:", ["Technical Round", "Interview Round"])
        
        if question_type == "Technical Round":
            technical_subtype = st.selectbox("Select Technical Round Type:", ["MCQs", "Coding Challenges"])
        
        if st.button("Generate Questions"):
            if question_type == "Technical Round":
                if technical_subtype == "MCQs":
                    quiz_response = generate_mcq_questions(st.session_state.analysis, st.session_state.selected_domain)
                else:
                    quiz_response = generate_coding_questions(st.session_state.analysis, st.session_state.selected_domain)
            else:
                quiz_response = generate_interview_questions(st.session_state.analysis, st.session_state.selected_domain)
            
            questions, answers = split_questions_answers(quiz_response)
            
            st.write("Generated Questions:")
            st.write(questions)
            
            if st.button("Show Answers"):
                st.write("Answers:")
                st.write(answers)

    elif uploaded_file is not None and uploaded_file == st.session_state.uploaded_file:
        st.write("Analysis Result:")
        st.write(st.session_state.analysis)
        
        # Extract domains from the analysis
        domain_response = extract_domains_from_analysis(st.session_state.analysis)
        domains = [domain.strip() for domain in domain_response.split(',')]
        default_domains = ["DSA", "DBMS", "Programming Basics"] + domains

        def update_selected_domain():
            st.session_state.selected_domain = st.session_state.domain_select

        st.selectbox(
            "Select Domain for Questions:",
            default_domains,
            key="domain_select",
            index=default_domains.index(st.session_state.selected_domain) if st.session_state.selected_domain in default_domains else 0,
            on_change=update_selected_domain
        )
        
        question_type = st.selectbox("Select Question Type:", ["Technical Round", "Interview Round"])
        
        if question_type == "Technical Round":
            technical_subtype = st.selectbox("Select Technical Round Type:", ["MCQs", "Coding Challenges"])
        
        if st.button("Generate Questions"):
            if question_type == "Technical Round":
                if technical_subtype == "MCQs":
                    quiz_response = generate_mcq_questions(st.session_state.analysis, st.session_state.selected_domain)
                else:
                    quiz_response = generate_coding_questions(st.session_state.analysis, st.session_state.selected_domain)
            else:
                quiz_response = generate_interview_questions(st.session_state.analysis, st.session_state.selected_domain)
            
            questions, answers = split_questions_answers(quiz_response)
            
            st.write("Generated Questions:")
            st.write(questions)
            
            if st.button("Show Answers"):
                st.write("Answers:")
                st.write(answers)

if __name__ == "__main__":
    main()