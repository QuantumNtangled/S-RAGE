from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from evaluation.llm_provider import LLMProvider
import os
from dotenv import load_dotenv

class RAGEvaluator:
    def __init__(self, llm_provider: LLMProvider, embedding_provider: str = "azure"):
        load_dotenv()
        self.rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.llm = llm_provider
        
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

    async def calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        # Get embeddings
        embedding1 = await self.get_embedding(text1)
        embedding2 = await self.get_embedding(text2)
        
        # Reshape for sklearn cosine_similarity
        similarity = cosine_similarity(
            embedding1.reshape(1, -1), 
            embedding2.reshape(1, -1)
        )[0][0]
        
        return float(similarity)

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