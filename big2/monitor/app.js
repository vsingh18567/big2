/**
 * Big2 PPO Training Monitor
 * Real-time dashboard for monitoring training progress
 */

// Configuration
const POLL_INTERVAL = 1000; // 1 second
const MAX_CHART_POINTS = 500; // Limit chart points for performance

// Chart instances
let winRateChart = null;
let lossChart = null;
let entropyLrChart = null;
let opponentMixChart = null;

// State
let lastStats = null;
let pollTimer = null;
let connectionRetries = 0;

// Chart.js default configuration
Chart.defaults.color = '#8b949e';
Chart.defaults.borderColor = '#30363d';
Chart.defaults.font.family = "'Space Grotesk', -apple-system, sans-serif";

/**
 * Initialize the dashboard
 */
function init() {
    initCharts();
    startPolling();
}

/**
 * Initialize all charts
 */
function initCharts() {
    // Win Rate Chart
    const winRateCtx = document.getElementById('win-rate-chart').getContext('2d');
    winRateChart = new Chart(winRateCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'vs Greedy',
                    data: [],
                    borderColor: 'rgba(63, 185, 80, 1)',
                    backgroundColor: 'rgba(63, 185, 80, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                },
                {
                    label: 'vs Smart',
                    data: [],
                    borderColor: 'rgba(163, 113, 247, 1)',
                    backgroundColor: 'rgba(163, 113, 247, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 2,
                    pointHoverRadius: 5,
                },
                {
                    label: 'Random Baseline (25%)',
                    data: [],
                    borderColor: 'rgba(110, 118, 129, 0.5)',
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0,
                    pointRadius: 0,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 12,
                        padding: 15,
                    }
                },
                tooltip: {
                    backgroundColor: '#21262d',
                    titleColor: '#e6edf3',
                    bodyColor: '#8b949e',
                    borderColor: '#30363d',
                    borderWidth: 1,
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(1)}%`
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Batch',
                    },
                    grid: {
                        color: '#21262d',
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Win Rate',
                    },
                    min: 0,
                    max: 1,
                    ticks: {
                        callback: (value) => `${(value * 100).toFixed(0)}%`
                    },
                    grid: {
                        color: '#21262d',
                    }
                }
            }
        }
    });

    // Loss Chart
    const lossCtx = document.getElementById('loss-chart').getContext('2d');
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Policy Loss',
                    data: [],
                    borderColor: 'rgba(88, 166, 255, 1)',
                    backgroundColor: 'rgba(88, 166, 255, 0.1)',
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                },
                {
                    label: 'Value Loss',
                    data: [],
                    borderColor: 'rgba(240, 136, 62, 1)',
                    backgroundColor: 'rgba(240, 136, 62, 0.1)',
                    fill: false,
                    tension: 0.3,
                    pointRadius: 0,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 12,
                        padding: 15,
                    }
                },
                tooltip: {
                    backgroundColor: '#21262d',
                    titleColor: '#e6edf3',
                    bodyColor: '#8b949e',
                    borderColor: '#30363d',
                    borderWidth: 1,
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Batch',
                    },
                    grid: {
                        color: '#21262d',
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Loss',
                    },
                    grid: {
                        color: '#21262d',
                    }
                }
            }
        }
    });

    // Entropy & LR Chart
    const entropyLrCtx = document.getElementById('entropy-lr-chart').getContext('2d');
    entropyLrChart = new Chart(entropyLrCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Entropy',
                    data: [],
                    borderColor: 'rgba(210, 153, 34, 1)',
                    backgroundColor: 'rgba(210, 153, 34, 0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 0,
                    yAxisID: 'y',
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 12,
                        padding: 15,
                    }
                },
                tooltip: {
                    backgroundColor: '#21262d',
                    titleColor: '#e6edf3',
                    bodyColor: '#8b949e',
                    borderColor: '#30363d',
                    borderWidth: 1,
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Batch',
                    },
                    grid: {
                        color: '#21262d',
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Entropy',
                    },
                    grid: {
                        color: '#21262d',
                    }
                }
            }
        }
    });

    // Opponent Mix Chart (Doughnut)
    const opponentMixCtx = document.getElementById('opponent-mix-chart').getContext('2d');
    opponentMixChart = new Chart(opponentMixCtx, {
        type: 'doughnut',
        data: {
            labels: ['Greedy', 'Smart', 'Self-play', 'Random'],
            datasets: [{
                data: [25, 25, 25, 25],
                backgroundColor: [
                    'rgba(63, 185, 80, 0.8)',
                    'rgba(163, 113, 247, 0.8)',
                    'rgba(88, 166, 255, 0.8)',
                    'rgba(110, 118, 129, 0.8)',
                ],
                borderColor: '#1c2128',
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        boxWidth: 12,
                        padding: 10,
                    }
                },
                tooltip: {
                    backgroundColor: '#21262d',
                    titleColor: '#e6edf3',
                    bodyColor: '#8b949e',
                    borderColor: '#30363d',
                    borderWidth: 1,
                    callbacks: {
                        label: (ctx) => `${ctx.label}: ${ctx.raw.toFixed(1)}%`
                    }
                }
            }
        }
    });
}

/**
 * Start polling for stats
 */
function startPolling() {
    fetchStats();
    pollTimer = setInterval(fetchStats, POLL_INTERVAL);
}

/**
 * Fetch training stats from the API
 */
async function fetchStats() {
    try {
        const response = await fetch('/api/monitor/stats');

        if (!response.ok) {
            if (response.status === 404) {
                // No training in progress
                updateStatus('idle', 'No training data');
                return;
            }
            throw new Error(`HTTP ${response.status}`);
        }

        const stats = await response.json();
        connectionRetries = 0;
        updateDashboard(stats);
        lastStats = stats;

    } catch (error) {
        connectionRetries++;
        if (connectionRetries > 5) {
            updateStatus('error', 'Connection lost');
        }
        console.warn('Failed to fetch stats:', error);
    }
}

/**
 * Update the entire dashboard with new stats
 */
function updateDashboard(stats) {
    updateStatus(stats.status, stats.status_message);
    updateProgress(stats);
    updateTimer(stats.elapsed_seconds);
    updateMetrics(stats);
    updateCharts(stats);
    updateInfoCards(stats);
    updateLastUpdate();
}

/**
 * Update status badge
 */
function updateStatus(status, message) {
    const badge = document.getElementById('status-badge');
    const statusText = badge.querySelector('.status-text');

    badge.className = `status-badge ${status}`;
    statusText.textContent = message || status.charAt(0).toUpperCase() + status.slice(1);
}

/**
 * Update progress bar and stats
 */
function updateProgress(stats) {
    const { current_batch, total_batches, elapsed_seconds } = stats;
    const progress = total_batches > 0 ? (current_batch / total_batches) * 100 : 0;

    document.getElementById('progress-bar').style.width = `${progress}%`;
    document.getElementById('progress-text').textContent = `${current_batch} / ${total_batches} batches`;
    document.getElementById('current-batch').textContent = current_batch.toLocaleString();

    // Calculate speed (batches per second)
    if (elapsed_seconds > 0 && current_batch > 0) {
        const speed = current_batch / elapsed_seconds;
        document.getElementById('batch-speed').textContent = `${speed.toFixed(2)} b/s`;

        // Calculate ETA
        const remainingBatches = total_batches - current_batch;
        const etaSeconds = remainingBatches / speed;
        document.getElementById('eta').textContent = formatTime(etaSeconds);
    }
}

/**
 * Update timer display
 */
function updateTimer(seconds) {
    document.getElementById('timer').textContent = formatTime(seconds);
}

/**
 * Update metric cards
 */
function updateMetrics(stats) {
    // Win rate vs greedy
    const wrGreedy = stats.win_rates_greedy;
    if (wrGreedy && wrGreedy.length > 0) {
        const latest = wrGreedy[wrGreedy.length - 1];
        document.getElementById('win-rate-greedy').textContent = `${(latest * 100).toFixed(1)}%`;

        if (wrGreedy.length > 1) {
            const prev = wrGreedy[wrGreedy.length - 2];
            const change = (latest - prev) * 100;
            const changeEl = document.getElementById('win-rate-greedy-change');
            changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(1)}% from last eval`;
            changeEl.className = `metric-subtitle ${change >= 0 ? 'positive' : 'negative'}`;
        }
    }

    // Win rate vs smart
    const wrSmart = stats.win_rates_smart;
    if (wrSmart && wrSmart.length > 0) {
        const latest = wrSmart[wrSmart.length - 1];
        document.getElementById('win-rate-smart').textContent = `${(latest * 100).toFixed(1)}%`;

        if (wrSmart.length > 1) {
            const prev = wrSmart[wrSmart.length - 2];
            const change = (latest - prev) * 100;
            const changeEl = document.getElementById('win-rate-smart-change');
            changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(1)}% from last eval`;
            changeEl.className = `metric-subtitle ${change >= 0 ? 'positive' : 'negative'}`;
        }
    }

    // Entropy
    const entropy = stats.entropy_history;
    if (entropy && entropy.length > 0) {
        const latest = entropy[entropy.length - 1];
        document.getElementById('entropy-value').textContent = latest.toFixed(3);
    }

    // Total loss
    const loss = stats.loss_history;
    if (loss && loss.length > 0) {
        const latest = loss[loss.length - 1];
        document.getElementById('total-loss').textContent = latest.toFixed(4);

        // Calculate trend (last 50 batches)
        if (loss.length > 50) {
            const recent = loss.slice(-50);
            const older = loss.slice(-100, -50);
            if (older.length > 0) {
                const recentAvg = recent.reduce((a, b) => a + b, 0) / recent.length;
                const olderAvg = older.reduce((a, b) => a + b, 0) / older.length;
                const trend = recentAvg - olderAvg;
                const trendEl = document.getElementById('loss-trend');
                trendEl.textContent = `${trend >= 0 ? '↑' : '↓'} ${Math.abs(trend).toFixed(4)} trend`;
                trendEl.className = `metric-subtitle ${trend <= 0 ? 'positive' : 'negative'}`;
            }
        }
    }
}

/**
 * Update charts with new data
 */
function updateCharts(stats) {
    // Downsample data if needed for performance
    const downsample = (arr, maxPoints) => {
        if (arr.length <= maxPoints) return arr;
        const step = Math.ceil(arr.length / maxPoints);
        return arr.filter((_, i) => i % step === 0);
    };

    // Win Rate Chart
    if (stats.eval_episodes && stats.eval_episodes.length > 0) {
        const labels = stats.eval_episodes;
        const baseline = labels.map(() => 0.25);

        winRateChart.data.labels = labels;
        winRateChart.data.datasets[0].data = stats.win_rates_greedy || [];
        winRateChart.data.datasets[1].data = stats.win_rates_smart || [];
        winRateChart.data.datasets[2].data = baseline;
        winRateChart.update('none');
    }

    // Loss Chart
    if (stats.policy_loss_history && stats.policy_loss_history.length > 0) {
        const policyLoss = downsample(stats.policy_loss_history, MAX_CHART_POINTS);
        const valueLoss = downsample(stats.value_loss_history || [], MAX_CHART_POINTS);
        const step = Math.ceil(stats.policy_loss_history.length / policyLoss.length);
        const labels = policyLoss.map((_, i) => (i + 1) * step);

        lossChart.data.labels = labels;
        lossChart.data.datasets[0].data = policyLoss;
        lossChart.data.datasets[1].data = valueLoss;
        lossChart.update('none');
    }

    // Entropy Chart
    if (stats.entropy_history && stats.entropy_history.length > 0) {
        const entropy = downsample(stats.entropy_history, MAX_CHART_POINTS);
        const step = Math.ceil(stats.entropy_history.length / entropy.length);
        const labels = entropy.map((_, i) => (i + 1) * step);

        entropyLrChart.data.labels = labels;
        entropyLrChart.data.datasets[0].data = entropy;
        entropyLrChart.update('none');
    }

    // Opponent Mix Chart
    if (stats.opponent_mix && Object.keys(stats.opponent_mix).length > 0) {
        const mix = stats.opponent_mix;
        opponentMixChart.data.datasets[0].data = [
            (mix.greedy || 0) * 100,
            (mix.smart || 0) * 100,
            (mix.self_play || 0) * 100,
            (mix.random || 0) * 100,
        ];
        opponentMixChart.update('none');
    }
}

/**
 * Update info cards
 */
function updateInfoCards(stats) {
    // Training parameters
    const config = stats.config_summary || {};
    document.getElementById('lr-value').textContent = stats.current_lr?.toExponential(2) || '--';
    document.getElementById('entropy-beta').textContent = stats.current_entropy_beta?.toFixed(4) || '--';
    document.getElementById('ppo-epochs').textContent = config.ppo_epochs || '--';
    document.getElementById('clip-epsilon').textContent = config.clip_epsilon || '--';
    document.getElementById('episodes-per-batch').textContent = config.episodes_per_batch || '--';
    document.getElementById('mini-batch-size').textContent = config.mini_batch_size || '--';

    // Curriculum state
    document.getElementById('num-checkpoints').textContent = stats.num_checkpoints || 0;
    document.getElementById('ema-greedy').textContent =
        stats.ema_win_rate_greedy != null ? `${(stats.ema_win_rate_greedy * 100).toFixed(1)}%` : '--';
    document.getElementById('ema-smart').textContent =
        stats.ema_win_rate_smart != null ? `${(stats.ema_win_rate_smart * 100).toFixed(1)}%` : '--';
    document.getElementById('mastery-greedy').textContent =
        stats.mastery_greedy != null ? `${(stats.mastery_greedy * 100).toFixed(0)}%` : '--';
    document.getElementById('mastery-smart').textContent =
        stats.mastery_smart != null ? `${(stats.mastery_smart * 100).toFixed(0)}%` : '--';
    document.getElementById('steps-per-episode').textContent =
        stats.steps_per_episode?.toFixed(1) || '--';
}

/**
 * Update last update timestamp
 */
function updateLastUpdate() {
    const now = new Date();
    document.getElementById('last-update').textContent = now.toLocaleTimeString();
}

/**
 * Format seconds to HH:MM:SS
 */
function formatTime(seconds) {
    if (!seconds || seconds < 0) return '--:--:--';

    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    return [
        hrs.toString().padStart(2, '0'),
        mins.toString().padStart(2, '0'),
        secs.toString().padStart(2, '0'),
    ].join(':');
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);

