from typing import List, Dict
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from evaluation.llm_provider import LLMProvider
import os
from dotenv import load_dotenv
class RAGEvaluator:

    def __init__(self, llm_provider: LLMProvider, embedding_provider: str = "azure"):
        load_dotenv()
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.llm = llm_provider
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.vectorizer = TfidfVectorizer()
        # Setup embedding model based on provider
        if embedding_provider == "azure":
            self.embedding_model = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        elif embedding_provider == "openai":
            self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL")
        elif embedding_provider == "sbert":
            sbert_model_name = os.getenv("SBERT_MODEL_NAME")
            self.model = SentenceTransformer(sbert_model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")
        
        self.embedding_provider = embedding_provider

    async def calculate_relevance(self, question: str, response: str) -> float:
        """Calculate how relevant the response is to the question."""
        prompt = f"""Rate the relevance of this response to the question on a scale of 0 to 1:
        Question: {question}
        Response: {response}
        Only return the numerical score, nothing else."""
        
        score = float((await self.llm.generate_completion(prompt)).strip())
        return min(max(score, 0), 1)  # Ensure score is between 0 and 1

    async def calculate_completeness(self, ground_truth: str, response: str) -> float:
        """Calculate how complete the response is compared to ground truth."""
        prompt = f"""Rate the completeness of this response compared to the ground truth on a scale of 0 to 1:
        Ground Truth: {ground_truth}
        Response: {response}
        Only return the numerical score, nothing else."""
        
        score = float((await self.llm.generate_completion(prompt)).strip())
        return min(max(score, 0), 1)

    async def calculate_consistency(self, ground_truth: str, response: str) -> float:
        """Calculate how consistent the response is with the ground truth."""
        prompt = f"""Rate the consistency of this response with the ground truth on a scale of 0 to 1:
        Ground Truth: {ground_truth}
        Response: {response}
        Only return the numerical score, nothing else."""
        
        score = float((await self.llm.generate_completion(prompt)).strip())
        return min(max(score, 0), 1)

    async def calculate_fluency(self, response: str) -> float:
        """Calculate the fluency of the response."""
        prompt = f"""Rate the fluency of this text on a scale of 0 to 1:
        Text: {response}
        Only return the numerical score, nothing else."""
        
        score = float((await self.llm.generate_completion(prompt)).strip())
        return min(max(score, 0), 1)

    def calculate_rouge_scores(self, candidate: str, reference: str) -> Dict:
        """Calculate ROUGE scores between candidate and reference texts."""
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts using embeddings."""
        try:
            # Create embeddings synchronously
            response1 = self.llm.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text1
            )
            response2 = self.llm.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text2
            )

            # Extract embedding vectors
            embedding1 = response1.data[0].embedding
            embedding2 = response2.data[0].embedding

            # Convert to numpy arrays for cosine similarity calculation
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
        except Exception as e:
            print(f"Error calculating cosine similarity: {str(e)}")
            return 0.0

    async def evaluate_chunk_completeness(self, chunk: str, ground_truth: str) -> float:
        """Evaluate how complete a chunk is compared to ground truth."""
        prompt = f"""Rate how much of the ground truth information is contained in this chunk on a scale of 0 to 1:
        Ground Truth: {ground_truth}
        Chunk: {chunk}
        Only return the numerical score, nothing else."""
        
        score = float((await self.llm.generate_completion(prompt)).strip())
        return min(max(score, 0), 1)

    async def evaluate_response_with_ai(self, question: str, ground_truth: str, response: str) -> str:
        """Get a qualitative evaluation of the response from the LLM."""
        prompt = f"""Evaluate this response based on the question and ground truth:
        Question: {question}
        Ground Truth: {ground_truth}
        Response: {response}
        
        Provide a brief evaluation focusing on:
        1. Accuracy
        2. Completeness
        3. Relevance
        Keep the evaluation concise."""
        
        return (await self.llm.generate_completion(prompt)).strip()
    
    
        
    async def get_embedding(self, text: str) -> np.ndarray:
        if self.embedding_provider == "sbert":
            return self.model.encode([text])[0]
        else:
            # Use the LLM provider's client for embeddings
            response = await self.llm.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return np.array(response.data[0].embedding)

    def calculate_rouge_scores(self, hypothesis: str, reference: str) -> Dict[str, float]:
        scores = self.rouge.score(reference, hypothesis)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    async def evaluate_response_with_ai(
        self, 
        question: str, 
        ground_truth: str, 
        response: str
    ) -> Dict[str, int]:
        system_prompt = """Task: Perform a comprehensive evaluation of the RAG (Retrieval-Augmented Generation) system's generated response in comparison to the Ground Truth answer.

Evaluation Criteria:
- Completeness: Assess whether the generated response covers all aspects of the Ground Truth answer. Point value: 2
- Accuracy: Check if the facts presented in the generated response align with the Ground Truth. Point value: 2
- Relevance: Determine the relevance of the generated content to the original question. Point value: 2
- Clarity: Evaluate the clarity and coherence of the generated response. Point value: 2
- Conciseness: Analyze if the response is succinct without losing essential details. Point value: 2

Using your instructions, provide a rating of 1-10 using the Evaluation Criteria scoring rubric. Only respond in an Integer between 1 and 10"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Question: {question}
Ground Truth: {ground_truth}
Generated Response: {response}"""}
        ]

        try:
            score = int(await self.llm.generate_completion(messages))
            return score
        except ValueError:
            return 0

    async def evaluate_chunk_completeness(
        self, 
        chunk: str, 
        ground_truth: str
    ) -> int:
        system_prompt = """Please evaluate the following chunk based on its completeness in relation to the Ground Truth answer. Determine the percentage of the Ground Truth answer that is contained within the chunk. Score the completeness on a scale of 1 to 10, where:

1: No relevant information is present.
2: Very minimal relevant information (1-10%).
3: Limited relevant information (11-25%).
4: Some relevant information (26-40%).
5: Moderate relevant information (41-55%).
6: Considerable relevant information (56-70%).
7: Good relevant information (71-85%).
8: Very good relevant information (86-95%).
9: Almost complete information (96-99%).
10: Complete and comprehensive information (100%).

Respond only with the integer score."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Ground Truth: {ground_truth}
Chunk: {chunk}"""}
        ]

        try:
            score = int(await self.llm.generate_completion(messages))
            return score
        except ValueError:
            return 0