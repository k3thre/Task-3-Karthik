# AI Recommendation Logic — Tech Stack & Career Recommender

An intelligent recommendation system that uses mathematical text processing and vector alignment to match a user's skills and interests with optimal career roles. This project demonstrates how to bypass complex black-box neural networks in favor of deterministic, high-precision similarity metrics.

---

## 🚀 System Architecture

The recommendation engine follows a clean **Input-Process-Output (IPO)** architecture:

1. **Input (Ingestion)**: Collects qualitative data directly from the user through an onboarding survey (minimum of three skills/interests).
2. **Process (Vector Space Alignment)**:
   * **TF-IDF Weighting**: Automatically penalizes generic high-frequency words and rewards highly descriptive, specific technical terms.
   * **Cosine Similarity**: Calculates the angular distance between the generated user profile vector and the predefined role vectors to measure exact alignment.
3. **Output (Filtering & Ranking)**: Sorts all matching roles in descending order and returns a refined **Top-3 tailored recommendation list**.

---

## 🛠️ Project Setup & Installation

### Prerequisites
Ensure you have **Python 3.x** installed on your system. 

### Dependencies
Install the required data processing and machine learning libraries using your terminal:

```bash
py -m pip install pandas scikit-learn
