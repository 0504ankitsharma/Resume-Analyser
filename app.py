import streamlit as st
import requests
import fitz
from PyPDF2 import PdfReader
from docx import Document
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables and configure Gemini API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def extract_text_from_pdf(file):
    """Extract text from PDF."""
    text = ""
    if isinstance(file, (str, bytes)):  # If file is a file path or bytes
        doc = fitz.open(file)
    else:  # If file is a file-like object (e.g., from st.file_uploader)
        doc = fitz.open(stream=file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_docx(docx_file):
    doc = Document(docx_file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
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
        "3. A overall explanation for each domain in brief according to the resume."
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

def analyze_documents(resume_text, job_description):
    custom_prompt = f"""
    Please analyze the following resume in the context of the job description provided. Strictly check every single line in job description and analyze my resume whether there is a match exactly. Strictly maintain high ATS standards and give scores only to the correct ones. Focus on hard skills which are missing and also soft skills which are missing. Provide the following details.:
    1. The match percentage of the resume to the job description. Display this.
    2. A list of missing keywords accurate ones.
    3. Final thoughts on the resume's overall match with the job description in 3 lines.
    4. Recommendations on how to add the missing keywords and improve the resume in 3-4 points with examples.
    Please display in the above order don't mention the numbers like 1. 2. etc and strictly follow ATS standards so that analysis will be accurate. Strictly follow the above templates omg. don't keep changing every time.
    Strictly follow the above things and template which has to be displayed and don't keep changing again and again. Don't fucking change the template from above.
    Title should be Resume analysis and maintain the same title for all. Also if someone uploads the same unchanged resume twice, keep in mind to give the same results. Display new ones only if they have changed their resume according to your suggestions or at least few changes.
    Job Description: {job_description}
    Resume: {resume_text}
    """
    return get_gemini_response(custom_prompt)

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
    num_questions = 20
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
    num_questions = 20
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
    num_questions = 20
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

def rephrase_text(text):
    custom_prompt = f"""
    Please rephrase the following text according to ATS standards, including quantifiable measures and improvements where possible, also maintain precise and concise points which will pass ATS screening:
    The title should be Rephrased Text:, and then display the output.
    Original Text: {text}
    """
    return get_gemini_response(custom_prompt)

def display_resume(file):
    file_type = file.name.split('.')[-1].lower()
    if file_type == 'pdf':
        text = extract_text_from_pdf(file)
    elif file_type == 'docx':
        text = extract_text_from_docx(file)
    else:
        st.error("Unsupported file type. Please upload a PDF or DOCX file.")
        return
    st.text_area("Parsed Resume Content", text, height=400)

def main():
    st.set_page_config(page_title="ATS Resume Evaluation System", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Resume Analyzer", "Question Generator", "Magic Write", "ATS Templates"])

    if page == "Resume Analyzer":
        st.title("📄🔍 ATS Resume Evaluation System")
        st.write("Welcome to the ATS Resume Evaluation System! Upload your resume and enter the job description to get a detailed evaluation of your resume's match with the job requirements.")

        job_description = st.text_area("Job Description:")
        resume = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])

        if resume:
            st.write("Uploaded Resume:")
            display_resume(resume)

        if st.button("Analyze Resume"):
            if job_description and resume:
                with st.spinner("Analyzing..."):
                    resume.seek(0)  # Reset the file pointer to the start
                    file_type = resume.name.split('.')[-1].lower()
                    if file_type == 'pdf':
                        resume_text = extract_text_from_pdf(resume)
                    elif file_type == 'docx':
                        resume_text = extract_text_from_docx(resume)
                    
                    analysis = analyze_documents(resume_text, job_description)
                    st.session_state.analysis = analysis  # Store analysis in session state
                    st.markdown(analysis)
                    
                    # Extract match percentage
                    lines = analysis.split("\n")
                    for line in lines:
                        if "match percentage" in line.lower():
                            match_percentage = line.split(":")[-1].strip()
                            match_percentage = ''.join(filter(str.isdigit, match_percentage))
                            match_percentage = int(match_percentage)
                            break

                    st.write(f"Your Resume Match Percentage: {match_percentage}%")
                    st.progress(match_percentage)

                st.success("Analysis Complete!")
            else:
                st.error("Please enter the job description and upload a resume.")

    elif page == "Question Generator":
        st.title("❓ Question Generator")
        st.write("Generate questions based on your resume analysis.")

        if 'analysis' not in st.session_state:
            st.warning("Please analyze your resume first in the Resume Analyzer section.")
            return

        domain_response = extract_domains_from_analysis(st.session_state.analysis)
        domains = [domain.strip() for domain in domain_response.split(',')]
        default_domains = ["DSA", "DBMS", "Programming Basics"] + domains

        selected_domain = st.selectbox("Select Domain for Questions:", default_domains)
        
        question_type = st.selectbox("Select Question Type:", ["Technical Round", "Interview Round"])
        
        if question_type == "Technical Round":
            technical_subtype = st.selectbox("Select Technical Round Type:", ["MCQs", "Coding Challenges"])
        
        if st.button("Generate Questions"):
            with st.spinner("Generating questions..."):
                if question_type == "Technical Round":
                    if technical_subtype == "MCQs":
                        quiz_response = generate_mcq_questions(st.session_state.analysis, selected_domain)
                    else:
                        quiz_response = generate_coding_questions(st.session_state.analysis, selected_domain)
                else:
                    quiz_response = generate_interview_questions(st.session_state.analysis, selected_domain)
                
                questions, answers = split_questions_answers(quiz_response)
                
                st.write("Generated Questions:")
                st.write(questions)
                
                if st.button("Show Answers"):
                    st.write("Answers:")
                    st.write(answers)

    elif page == "Magic Write":
        st.title("🔮 Magic Write")
        st.write("Enter lines from your resume to rephrase them according to ATS standards with quantifiable measures.")

        text_to_rephrase = st.text_area("Text to Rephrase:")
        
        if st.button("Rephrase"):
            if text_to_rephrase:
                with st.spinner("Rephrasing..."):
                    rephrased_text = rephrase_text(text_to_rephrase)
                    st.write(rephrased_text)
                st.success("Rephrasing Complete!")
            else:
                st.error("Please enter the text you want to rephrase.")

    elif page == "ATS Templates":
        st.title("📄📝 Free ATS Resume Templates")
        st.write("Download free ATS-friendly resume templates. Click on a template to download it.")

        templates = {
            "Sample 1": "https://docs.google.com/document/d/1NWFIz-EZ1ZztZSdXfrrcdffSzG-uermd/edit?usp=sharing&ouid=102272826109592952279&rtpof=true&sd=true",
            "Sample 2": "https://docs.google.com/document/d/1xO7hvK-RQSb0mjXRn24ri3AiDrXx6qt8/edit?usp=sharing&ouid=102272826109592952279&rtpof=true&sd=true",
            "Sample 3": "https://docs.google.com/document/d/1fAukvT0lWXns3VexbZjwXyCAZGw2YptO/edit?usp=sharing&ouid=102272826109592952279&rtpof=true&sd=true",
            "Sample 4": "https://docs.google.com/document/d/1htdoqTPDnG-T0OpTtj8wUOIfX9PfvqhS/edit?usp=sharing&ouid=102272826109592952279&rtpof=true&sd=true",
            "Sample 5": "https://docs.google.com/document/d/1uTINCs71c4lL1Gcb8DQlyFYVqzOPidoS/edit?usp=sharing&ouid=102272826109592952279&rtpof=true&sd=true",
            "Sample 6": "https://docs.google.com/document/d/1KO9OuhY7l6dn2c5xynpCOIgbx5LWsfb0/edit?usp=sharing&ouid=102272826109592952279&rtpof=true&sd=true"
        }

        cols = st.columns(3)
        for index, (template_name, template_link) in enumerate(templates.items()):
            col = cols[index % 3]
            col.markdown(f"""
                <div style="text-align:center">
                    <iframe src="https://drive.google.com/file/d/{template_link.split('/')[-2]}/preview" width="200" height="250" allow="autoplay"></iframe>
                    <br>
                    <a href="{template_link}" target="_blank">{template_name}</a>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()