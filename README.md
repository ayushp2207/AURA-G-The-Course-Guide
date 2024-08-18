# Course Information Assistant

Welcome to the **Course Information Assistant**! This project allows you to interactively inquire about university courses using PDF files as the source of information. The system leverages AI to provide structured, clear, and engaging responses to your questions.

## Features

- **Interactive Q&A:** Ask questions about course names, faculty, schedules, teaching materials, evaluation components, and more.
- **AI-Powered Responses:** Utilizes Google Gemini API and FAISS vector store for accurate and context-based answers.
- **PDF Integration:** Extracts course details from PDF files stored in a designated folder.
- **Streamlit Interface:** Easy-to-use web interface for seamless interaction.

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

    Place your course PDFs in the `pdfs` folder. Ensure the filenames are descriptive of the course (e.g., `data_science.pdf`).

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, feel free to open an issue on GitHub or contact the maintainer.

---

This README will guide others in using your project effectively and contribute to it if they wish.
