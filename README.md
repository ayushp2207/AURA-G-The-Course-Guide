# Course Information Assistant

Certainly. Here's an edited version of the README content that incorporates information about RAG without referring to specific code:

Course Information Assistant
Welcome to the **Course Information Assistant**! This project allows you to interactively inquire about Ahmedabad University's courses. The system leverages AI and Retrieval-Augmented Generation (RAG) to provide structured, clear, and engaging responses to your questions.

# Features
* **Interactive Q&A:** Ask questions about course names, faculty, schedules, teaching materials, evaluation components, and more.
* **AI-Powered Responses:** Utilizes advanced language models and RAG for accurate and context-based answers.
* **Streamlit Interface:** Easy-to-use web interface for seamless interaction.

Retrieval-Augmented Generation (RAG)
This project implements RAG to enhance the quality and accuracy of responses:

1. **Retrieval:** The system indexes course information from PDF files and retrieves relevant content based on user queries.
2. **Augmentation:** Retrieved information is used to provide context to the AI model.
3. **Generation:** An AI model generates natural language responses, combining its knowledge with the retrieved course-specific information.

By using RAG, the Course Information Assistant can provide more accurate, contextually relevant, and up-to-date answers about specific courses, enhancing the overall user experience.

## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/course-information-assistant.git
    cd course-information-assistant
    ```

2. **Install the Required Packages:**

    Make sure you have Python installed, then run:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables:**

    Create a `.env` file in the root directory of the project and add your Google API key:

    ```
    GOOGLE_API_KEY=your-google-api-key
    ```

    You can also store the API key using Streamlit Secrets:

    ```python
    # In Streamlit Cloud, add the secret directly in the UI
    st.secrets["GOOGLE_API_KEY"] = "your-google-api-key"
    ```

4. **Add PDF Files:**

    Place your course PDFs in the `pdfs` folder. Ensure the filenames contain course code followed by name of the course (e.g., `CSD102 - Data Science.pdf`).

## Usage

1. **Run the Application:**

    Start the Streamlit server:

    ```bash
    streamlit run app.py
    ```

2. **Interact with the Assistant:**

    - Enter the course name to search for the corresponding PDF.
    - Ask questions about the course details, and the assistant will provide responses based on the content of the PDF.

3. **Example Questions:**

    - "Who is the faculty for the course?"
    - "What is the course schedule?"
    - "What materials will be used for teaching?"

## Contributing

Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.

## Contact

For any questions or issues, feel free to open an issue on GitHub or contact the maintainer.

---

This README will guide others in using your project effectively and contribute to it if they wish.
