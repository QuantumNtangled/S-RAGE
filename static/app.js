document.addEventListener('DOMContentLoaded', function() {
    const loading = document.getElementById('loading');
    const resultsList = document.getElementById('resultsList');

    async function loadResults() {
        try {
            loading.style.display = 'block';
            
            const response = await fetch('/api/results');
            const results = await response.json();
            
            resultsList.innerHTML = results.map(result => {
                // Safely parse chunks and evaluation
                let chunksHtml = '';
                let evaluationHtml = '';
                
                try {
                    const chunks = Array.isArray(result.chunks) ? result.chunks : JSON.parse(result.chunks || '[]');
                    chunksHtml = chunks.map((chunk, index) => `
                        <div class="chunk">
                            <h4>Chunk ${index + 1}</h4>
                            <pre>${chunk}</pre>
                        </div>
                    `).join('');
                } catch (e) {
                    console.error('Error parsing chunks:', e);
                    chunksHtml = '<p>Error displaying chunks</p>';
                }
                
                try {
                    const evaluation = result.evaluation ? (typeof result.evaluation === 'string' ? JSON.parse(result.evaluation) : result.evaluation) : null;
                    if (evaluation) {
                        const responseEval = evaluation.response_evaluation;
                        const chunksEval = evaluation.chunks_evaluation;
                        
                        evaluationHtml = `
                            <div class="evaluation">
                                <h3>Evaluation Metrics</h3>
                                <div class="metrics-container">
                                    <div class="metrics-grid">
                                        <div class="metric-card">
                                            <h4>Response Metrics</h4>
                                            <h5>Relevance: ${responseEval.relevance}</h5>
                                            <h5>Completeness: ${responseEval.completeness}</h5>
                                            <h5>Consistency: ${responseEval.consistency}</h5>
                                            <h5>Fluency: ${responseEval.fluency}</h5>
                                            <h5>Cosine Similarity: ${responseEval.cosine_similarity}</h5>
                                            <div class="rouge-scores">
                                                <h5>ROUGE Scores:</h5>
                                                <p>ROUGE-1: ${responseEval.rouge_scores.rouge1}</p>
                                                <p>ROUGE-2: ${responseEval.rouge_scores.rouge2}</p>
                                                <p>ROUGE-L: ${responseEval.rouge_scores.rougeL}</p>
                                            </div>
                                        </div>
                                        <div class="metric-card">
                                            <h4>AI Evaluation</h4>
                                            <pre>${responseEval.ai_evaluation}</pre>
                                        </div>
                                    </div>
                                    ${chunksEval.length > 0 ? `
                                        <div class="chunk-metrics">
                                            <h4>Chunks Evaluation</h4>
                                            ${chunksEval.map((chunkEval, index) => `
                                                <div class="metric-card">
                                                    <h5>Chunk ${index + 1}</h5>
                                                    <p>Cosine Similarity: ${chunkEval.cosine_similarity}</p>
                                                    <p>Completeness: ${chunkEval.completeness}</p>
                                                    <div class="rouge-scores">
                                                        <h5>ROUGE Scores:</h5>
                                                        <p>ROUGE-1: ${chunkEval.rouge_scores.rouge1}</p>
                                                        <p>ROUGE-2: ${chunkEval.rouge_scores.rouge2}</p>
                                                        <p>ROUGE-L: ${chunkEval.rouge_scores.rougeL}</p>
                                                    </div>
                                                </div>
                                            `).join('')}
                                        </div>
                                    ` : ''}
                                </div>
                            </div>
                        `;
                    } else {
                        evaluationHtml = '<p>No evaluation data available</p>';
                    }
                } catch (e) {
                    console.error('Error parsing evaluation:', e);
                    evaluationHtml = '<p>Error displaying evaluation</p>';
                }

                return `
                    <div class="result">
                        <h3>Question</h3>
                        <p>${result.question}</p>
                        
                        <h3>Ground Truth</h3>
                        <p>${result.ground_truth}</p>
                        
                        <h3>Response</h3>
                        <p>${result.response}</p>
                        
                        <h3>Chunks</h3>
                        <div class="chunks-container">
                            ${chunksHtml}
                        </div>
                        
                        ${evaluationHtml}
                        
                        <p class="timestamp">Timestamp: ${result.timestamp}</p>
                    </div>
                `;
            }).join('');
            
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while loading results');
        } finally {
            loading.style.display = 'none';
        }
    }

    // Refresh results every 30 seconds
    setInterval(loadResults, 30000);

    // Initial load
    document.addEventListener('DOMContentLoaded', loadResults);
}); 