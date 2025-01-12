document.addEventListener('DOMContentLoaded', function() {
    const loading = document.getElementById('loading');
    const resultsList = document.getElementById('resultsList');

    async function loadResults() {
        try {
            loading.style.display = 'block';
            
            const response = await fetch('/api/results');
            const results = await response.json();
            
            resultsList.innerHTML = results.map(result => `
                <div class="result-item">
                    <h3>Question:</h3>
                    <p>${result.question}</p>
                    
                    <h3>Ground Truth:</h3>
                    <p>${result.ground_truth}</p>
                    
                    <h3>RAG Response:</h3>
                    <p>${result.response || 'Not processed yet'}</p>
                    
                    <div class="metrics-container">
                        <h3>Evaluation Metrics:</h3>
                        ${result.evaluation ? `
                            <div class="metrics-grid">
                                <div class="metric-card">
                                    <h4>Response Metrics</h4>
                                    <p>Cosine Similarity: ${result.evaluation.response_metrics.cosine_similarity.toFixed(3)}</p>
                                    <p>AI Evaluation Score: ${result.evaluation.response_metrics.ai_evaluation}/10</p>
                                    <div class="rouge-scores">
                                        <h5>ROUGE Scores:</h5>
                                        <p>ROUGE-1: ${result.evaluation.response_metrics.rouge_scores.rouge1.toFixed(3)}</p>
                                        <p>ROUGE-2: ${result.evaluation.response_metrics.rouge_scores.rouge2.toFixed(3)}</p>
                                        <p>ROUGE-L: ${result.evaluation.response_metrics.rouge_scores.rougeL.toFixed(3)}</p>
                                    </div>
                                </div>
                                
                                <div class="metric-card">
                                    <h4>Chunks Evaluation</h4>
                                    ${result.evaluation.chunks_evaluation.map((chunk, index) => `
                                        <div class="chunk-metrics">
                                            <h5>Chunk ${index + 1}</h5>
                                            <p>Cosine Similarity: ${chunk.cosine_similarity.toFixed(3)}</p>
                                            <p>Completeness Score: ${chunk.completeness}/10</p>
                                            <div class="rouge-scores">
                                                <p>ROUGE-1: ${chunk.rouge_scores.rouge1.toFixed(3)}</p>
                                                <p>ROUGE-2: ${chunk.rouge_scores.rouge2.toFixed(3)}</p>
                                                <p>ROUGE-L: ${chunk.rouge_scores.rougeL.toFixed(3)}</p>
                                            </div>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        ` : '<p>No evaluation data available</p>'}
                    </div>
                    
                    <h3>Chunks:</h3>
                    <div class="chunks">
                        <pre>${result.chunks ? JSON.parse(result.chunks).join('\n\n') : 'No chunks available'}</pre>
                    </div>
                    
                    <small>Processed: ${result.timestamp || 'N/A'}</small>
                </div>
            `).join('');
            
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while loading results');
        } finally {
            loading.style.display = 'none';
        }
    }

    // Automatically load results on page load
    loadResults();
}); 