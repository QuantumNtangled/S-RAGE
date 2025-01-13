from typing import List, Dict
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from evaluation.llm_provider import LLMProvider
import os
from dotenv import load_dotenv
from bert_score import score
import torch
import nltk
from nltk.tokenize import sent_tokenize
import re
nltk.download('punkt')

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
        # Set device for BERT Score
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = 500  # Maximum number of tokens per chunk

    async def calculate_relevance(self, question: str, response: str) -> float:
        """Calculate how relevant the response is to the question."""
        prompt = f"""Rate the relevance of this response to the question on a scale of 0 to 1.

Relevance Rubric:
1.0: Perfect relevance - Response directly and comprehensively addresses the question
0.8: High relevance - Response addresses the main points but might miss minor details
0.6: Moderate relevance - Response is on topic but misses some key points or includes unnecessary information
0.4: Low relevance - Response only tangentially relates to the question
0.2: Very low relevance - Response barely addresses the question
0.0: No relevance - Response does not address the question at all

Question: {question}
Response: {response}

Provide only a number between 0 and 1 with 2 decimal places (e.g., 0.75)."""

        try:
            score = float((await self.llm.generate_completion(prompt)).strip())
            return round(min(max(score, 0), 1), 2)
        except Exception as e:
            print(f"Error calculating relevance: {str(e)}")
            return 0.0

    async def calculate_completeness(self, ground_truth: str, response: str) -> float:
        """Calculate how complete the response is compared to ground truth."""
        prompt = f"""Rate the completeness of this response compared to the ground truth on a scale of 0 to 1.

Completeness Rubric:
1.0: Complete coverage - All key points and supporting details from ground truth are present
0.8: High coverage - Most key points present with minor details missing
0.6: Moderate coverage - Main points covered but significant details missing
0.4: Partial coverage - Some key points missing, incomplete explanation
0.2: Minimal coverage - Only basic or surface-level information included
0.0: No coverage - None of the ground truth information is present

Ground Truth: {ground_truth}
Response: {response}

Provide only a number between 0 and 1 with 2 decimal places (e.g., 0.75)."""

        try:
            score = float((await self.llm.generate_completion(prompt)).strip())
            return round(min(max(score, 0), 1), 2)
        except Exception as e:
            print(f"Error calculating completeness: {str(e)}")
            return 0.0

    async def calculate_consistency(self, ground_truth: str, response: str) -> float:
        """Calculate how consistent the response is with the ground truth."""
        prompt = f"""Rate the consistency of this response with the ground truth on a scale of 0 to 1.

Consistency Rubric:
1.0: Perfect consistency - No contradictions, all information aligns with ground truth
0.8: High consistency - Minor discrepancies but no significant contradictions
0.6: Moderate consistency - Some contradictions present but major points align
0.4: Low consistency - Several contradictions or misaligned information
0.2: Very low consistency - Mostly contradicts ground truth
0.0: No consistency - Completely contradicts ground truth or provides entirely incorrect information

Ground Truth: {ground_truth}
Response: {response}

Provide only a number between 0 and 1 with 2 decimal places (e.g., 0.75)."""

        try:
            score = float((await self.llm.generate_completion(prompt)).strip())
            return round(min(max(score, 0), 1), 2)
        except Exception as e:
            print(f"Error calculating consistency: {str(e)}")
            return 0.0

    async def calculate_fluency(self, response: str) -> float:
        """Calculate the fluency of the response."""
        prompt = f"""Rate the fluency of this text on a scale of 0 to 1.

Fluency Rubric:
1.0: Perfect fluency - Professional-level writing, clear structure, excellent flow
0.8: High fluency - Well-written, minor stylistic issues
0.6: Moderate fluency - Generally clear but with some awkward phrasing or structure
0.4: Low fluency - Frequent awkward phrasing, grammar issues, or poor structure
0.2: Very low fluency - Major readability issues, confusing structure
0.0: No fluency - Incomprehensible or severely broken language

Text: {response}

Provide only a number between 0 and 1 with 2 decimal places (e.g., 0.75)."""

        try:
            score = float((await self.llm.generate_completion(prompt)).strip())
            return round(min(max(score, 0), 1), 2)
        except Exception as e:
            print(f"Error calculating fluency: {str(e)}")
            return 0.0

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

    async def evaluate_response_with_ai(
        self, 
        question: str, 
        ground_truth: str, 
        response: str
    ) -> int:
        """Get a numerical evaluation score from 1-10."""
        system_prompt = """Task: Evaluate this RAG (Retrieval-Augmented Generation) response on a scale of 1-10.

Evaluation Criteria (2 points each):
1. Completeness: Does the response cover all key aspects of the Ground Truth?
2. Accuracy: Are the facts presented correct and aligned with the Ground Truth?
3. Relevance: Does the response directly address the original question?
4. Clarity: Is the response well-structured and easy to understand?
5. Conciseness: Is the response focused without unnecessary information?

Scoring Guide:
10: Exceptional - Perfect in all criteria
8-9: Excellent - Minor flaws in 1-2 areas
6-7: Good - Some gaps but generally effective
4-5: Fair - Significant issues in multiple areas
2-3: Poor - Major problems throughout
1: Inadequate - Fails to meet any criteria effectively

Provide only the numerical score (1-10) as your response."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"""Question: {question}
Ground Truth: {ground_truth}
Generated Response: {response}"""}
        ]

        try:
            print("Generating AI evaluation score...")  # Debug log
            score = int(await self.llm.generate_completion(messages))
            print(f"AI evaluation score: {score}")  # Debug log
            
            # Ensure score is within valid range
            score = max(1, min(10, score))
            return score
        except ValueError as e:
            print(f"Error parsing AI evaluation score: {str(e)}")
            return 0
        except Exception as e:
            print(f"Error in AI evaluation: {str(e)}")
            return 0
    
    
        
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

    def chunk_text(self, text: str) -> list:
        """Split text into sentences and then combine into chunks under max_length."""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            # Rough estimation of tokens (words + punctuation)
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.max_length:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def calculate_bert_score(self, candidate: str, reference: str) -> Dict:
        """Calculate BERT score between candidate and reference texts, handling long texts."""
        try:
            print(f"Calculating BERT score for texts of lengths: {len(candidate)}, {len(reference)}")
            
            # Split both texts into chunks
            candidate_chunks = self.chunk_text(candidate)
            reference_chunks = self.chunk_text(reference)
            
            print(f"Split into {len(candidate_chunks)} candidate chunks and {len(reference_chunks)} reference chunks")
            
            # Calculate scores for each combination of chunks
            precisions = []
            recalls = []
            f1s = []
            
            for cand_chunk in candidate_chunks:
                for ref_chunk in reference_chunks:
                    P, R, F1 = score([cand_chunk], [ref_chunk], lang='en', device=self.device)
                    precisions.append(float(P[0]))
                    recalls.append(float(R[0]))
                    f1s.append(float(F1[0]))
            
            # Take the maximum scores (assuming each chunk should match its best counterpart)
            if precisions:
                result = {
                    'precision': max(precisions),
                    'recall': max(recalls),
                    'f1': max(f1s)
                }
                print(f"BERT scores: {result}")
                return result
            else:
                print("No valid chunks to score")
                return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                
        except Exception as e:
            print(f"Error calculating BERT score: {str(e)}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    def calculate_scores(self, candidate: str, reference: str) -> Dict:
        """Calculate BERT scores."""
        return self.calculate_bert_score(candidate, reference)

    def remove_markdown(self, text: str) -> str:
        """Remove markdown formatting from text."""
        if not isinstance(text, str):
            return ""
            
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Remove inline code
        text = re.sub(r'`[^`]*`', '', text)
        
        # Remove headers
        text = re.sub(r'#{1,6}\s.*?\n', '', text)
        
        # Remove bold/italic
        text = re.sub(r'\*\*.*?\*\*', '', text)
        text = re.sub(r'__.*?__', '', text)
        text = re.sub(r'\*.*?\*', '', text)
        text = re.sub(r'_.*?_', '', text)
        
        # Remove links
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove images
        text = re.sub(r'!\[([^\]]+)\]\([^\)]+\)', '', text)
        
        # Remove bullet points and numbered lists
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.strip()
        
        return text

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity using embeddings after removing markdown."""
        try:
            # Remove markdown from both texts
            clean_text1 = self.remove_markdown(text1)
            clean_text2 = self.remove_markdown(text2)
            
            print(f"Calculating similarity for texts of lengths: {len(clean_text1)}, {len(clean_text2)}")
            
            # Get embeddings for cleaned texts
            embedding1 = self.llm.client.embeddings.create(
                model="text-embedding-ada-002",
                input=clean_text1
            ).data[0].embedding

            embedding2 = self.llm.client.embeddings.create(
                model="text-embedding-ada-002",
                input=clean_text2
            ).data[0].embedding

            # Calculate cosine similarity
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            
            return round(float(similarity), 2)
        except Exception as e:
            print(f"Error calculating semantic similarity: {str(e)}")
            return 0.0