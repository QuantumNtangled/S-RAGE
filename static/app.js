document.addEventListener('DOMContentLoaded', function() {
    const loading = document.getElementById('loading');
    const resultsList = document.querySelector('.results');

    // Make evaluateResponse globally accessible
    window.evaluateResponse = async function(groundTruthId) {
        try {
            loading.style.display = 'block';
            const response = await fetch(`/api/evaluate/${groundTruthId}`, {
                method: 'POST'
            });
            const result = await response.json();
            console.log('Evaluation result:', result);
            loadResults(); // Reload the results to show the evaluation
        } catch (error) {
            console.error('Error evaluating response:', error);
        } finally {
            loading.style.display = 'none';
        }
    }

    function loadResults() {
        loading.style.display = 'block';
        fetch('/api/results')
            .then(response => response.json())
            .then(data => {
                resultsList.innerHTML = data.map(result => {
                    let chunksHtml = '';
                    let evaluationHtml = '';
                    
                    try {
                        // Parse chunks and ensure it's an array
                        let chunks = [];
                        if (typeof result.chunks === 'string') {
                            chunks = JSON.parse(result.chunks || '[]');
                        } else if (Array.isArray(result.chunks)) {
                            chunks = result.chunks;
                        }

                        // Function to strip HTML tags
                        const stripHtml = (html) => {
                            const tmp = document.createElement('div');
                            tmp.innerHTML = html;
                            return tmp.textContent || tmp.innerText || '';
                        };

                        // Generate HTML for all chunks within the container
                        chunksHtml = `
                            <div class="chunks-container">
                                ${chunks.map((chunk, index) => `
                                    <div class="chunk">
                                        <h4>Chunk ${index + 1}</h4>
                                        <pre>${stripHtml(chunk)}</pre>
                                    </div>
                                `).join('')}
                            </div>
                        `;
                    } catch (e) {
                        console.error('Error parsing chunks:', e);
                        chunksHtml = '<div class="chunks-container"><p>Error displaying chunks</p></div>';
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
                                                <h5>Semantic Similarity: ${responseEval.semantic_similarity}</h5>
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
                                                        <p>Semantic Similarity: ${chunkEval.semantic_similarity}</p>
                                                        <p>Completeness: ${chunkEval.completeness}</p>
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

                    // Function to safely display text content
                    const safeText = (text) => {
                        const tmp = document.createElement('div');
                        tmp.textContent = text;
                        return tmp.innerHTML;
                    };

                    return `
                        <div class="result-item">
                            <h3>Question</h3>
                            <p>${safeText(result.question)}</p>
                            
                            <h3>Ground Truth</h3>
                            <pre class="ground-truth">${safeText(result.ground_truth)}</pre>
                            
                            <h3>Response</h3>
                            <pre class="response">${safeText(result.response)}</pre>
                            
                            <h3>Chunks</h3>
                            ${chunksHtml}
                            
                            <button onclick="evaluateResponse(${result.id})" class="evaluate-btn">
                                Evaluate Response
                            </button>
                            
                            ${evaluationHtml}
                        </div>
                    `;
                }).join('');
            })
            .catch(error => {
                console.error('Error:', error);
                resultsList.innerHTML = `<p>Error loading results: ${error.message}</p>`;
            })
            .finally(() => {
                loading.style.display = 'none';
            });
    }

    // Refresh results every 30 seconds
    setInterval(loadResults, 30000);

    // Initial load
    loadResults();
}); 