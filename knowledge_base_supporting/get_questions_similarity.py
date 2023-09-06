import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import get_config

questions_storage = get_config()['GenQuestions']['questions_storage']

similarity_threshold = get_config()['GenQuestionsSimilarity']['similarity_search_threshold']
simple_out = get_config()['GenQuestionsSimilarity']['simple_output']
similarity_storage = get_config()['GenQuestionsSimilarity']['similarity_storage']


def process_similarity():
    with open(questions_storage, 'r', encoding='utf-8') as file:
        questions_store = json.load(file)

    questions = []
    chunk_ids = []
    article_ids = []

    for chunk in questions_store['chunks']:
        chunk_id = chunk['id']
        article_id = chunk['article_id']
        for question in chunk['questions']:
            questions.append(question)
            chunk_ids.append(chunk_id)
            article_ids.append(article_id)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(questions)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    result_data = []

    for i, question in enumerate(questions):
        similar_questions = []
        for j, similarity in enumerate(cosine_sim[i]):
            if i != j and similarity > float(similarity_threshold):
                similar_questions.append({
                    "question": questions[j],
                    "chunk_id": chunk_ids[j],
                    "article_id": article_ids[j]
                })

        question_entry = {
            "question": question,
            "chunk_id": chunk_ids[i],
            "article_id": article_ids[i],
            "similar_questions": similar_questions
        }

        if bool(simple_out):
            question_entry["similarity"] = {
                "questions": len(similar_questions),
                "chunks": len(set(entry["chunk_id"] for entry in similar_questions)),
                "articles": len(set(entry["article_id"] for entry in similar_questions))
            }

        result_data.append(question_entry)

    if simple_out:
        result_data = sorted(result_data, key=lambda x: x["similarity"]["questions"], reverse=True)[:10]
        for entry in result_data:
            del entry["similar_questions"]

    if result_data:
        with open(similarity_storage, 'w', encoding='utf-8') as f:
            json.dump({"questions": result_data}, f, ensure_ascii=False, indent=4)
