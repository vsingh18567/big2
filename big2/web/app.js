// Game state
let currentGameId = null;
let selectedCards = [];
let legalMoves = [];

// Card suit and rank mappings
const SUITS = ["♦", "♣", "♥", "♠"];
const RANKS = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2"];
const SUIT_NAMES = ["diamonds", "clubs", "hearts", "spades"];

// Initialize event listeners
document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("start-btn").addEventListener("click", startGame);
    document.getElementById("new-game-btn").addEventListener("click", showSetup);
    document.getElementById("play-again-btn").addEventListener("click", showSetup);
    document.getElementById("play-btn").addEventListener("click", playSelectedCards);
    document.getElementById("pass-btn").addEventListener("click", playPass);
    document.getElementById("clear-selection-btn").addEventListener("click", clearSelection);
});

function showSetup() {
    document.getElementById("game-setup").classList.remove("hidden");
    document.getElementById("game-area").classList.add("hidden");
    document.getElementById("game-over").classList.add("hidden");
    currentGameId = null;
    selectedCards = [];
    legalMoves = [];
}

async function startGame() {
    const modelPath = document.getElementById("model-path").value.trim() || "big2_model.pt";
    const nPlayers = parseInt(document.getElementById("n-players").value);
    const device = document.getElementById("device").value;

    const startBtn = document.getElementById("start-btn");
    startBtn.disabled = true;
    startBtn.innerHTML = '<span class="loading"></span> Starting...';

    try {
        const response = await fetch("/api/game/start", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                model_path: modelPath,
                n_players: nPlayers,
                device: device,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to start game");
        }

        const gameState = await response.json();
        currentGameId = gameState.game_id;
        selectedCards = [];
        legalMoves = [];

        document.getElementById("game-setup").classList.add("hidden");
        document.getElementById("game-area").classList.remove("hidden");
        document.getElementById("game-over").classList.add("hidden");

        updateGameDisplay(gameState);
    } catch (error) {
        alert(`Error starting game: ${error.message}`);
        console.error(error);
    } finally {
        startBtn.disabled = false;
        startBtn.innerHTML = "Start Game";
    }
}

function updateGameDisplay(gameState) {
    // Update turn indicator
    const turnIndicator = document.getElementById("turn-indicator");
    if (gameState.is_human_turn) {
        turnIndicator.textContent = "🎮 Your Turn";
        turnIndicator.style.color = "#667eea";
    } else {
        turnIndicator.textContent = `⏳ Waiting for Player ${gameState.current_player}...`;
        turnIndicator.style.color = "#666";
    }

    // Update game status
    const gameStatus = document.getElementById("game-status");
    if (gameState.done) {
        gameStatus.textContent = gameState.is_human_winner
            ? "🎉 You Won!"
            : `Player ${gameState.winner} Won`;
        showGameOver(gameState);
        return;
    } else {
        gameStatus.textContent = `${gameState.passes_in_row} passes in a row`;
    }

    // Update hand
    updateHand(gameState.hand, gameState.hand_display, gameState.is_human_turn);
    document.getElementById("hand-count").textContent = gameState.hand_count;

    // Update trick
    updateTrick(gameState.trick);

    // Update opponents
    updateOpponents(gameState.opponents);

    // Update legal moves
    legalMoves = gameState.legal_moves || [];
    updateLegalMoves(legalMoves);

    // Update bot suggestion
    updateBotSuggestion(gameState.bot_suggestion, gameState.is_human_turn);

    // Update buttons
    const playBtn = document.getElementById("play-btn");
    const passBtn = document.getElementById("pass-btn");
    const clearBtn = document.getElementById("clear-selection-btn");

    if (gameState.is_human_turn) {
        playBtn.disabled = selectedCards.length === 0;
        passBtn.disabled = false;
        clearBtn.disabled = selectedCards.length === 0;
    } else {
        playBtn.disabled = true;
        passBtn.disabled = true;
        clearBtn.disabled = true;
    }
}

function updateHand(hand, handDisplay, isHumanTurn) {
    const handContainer = document.getElementById("hand");
    handContainer.innerHTML = "";

    hand.forEach((cardId, index) => {
        const card = createCardElement(cardId, handDisplay[index], isHumanTurn);
        handContainer.appendChild(card);
    });
}

function createCardElement(cardId, cardDisplay, clickable) {
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.cardId = cardId;

    if (!clickable) {
        card.classList.add("disabled");
    } else {
        card.addEventListener("click", () => toggleCardSelection(cardId, card));
    }

    const rank = cardDisplay[0];
    const suit = cardDisplay[1];
    const isRed = suit === "♥" || suit === "♦";
    const suitClass = isRed ? "suit-red" : "suit-black";

    card.innerHTML = `
        <div class="card-rank ${suitClass}">${rank}</div>
        <div class="card-suit ${suitClass}">${suit}</div>
        <div class="card-rank-bottom ${suitClass}">${rank}</div>
    `;

    return card;
}

function toggleCardSelection(cardId, cardElement) {
    const index = selectedCards.indexOf(cardId);
    if (index === -1) {
        selectedCards.push(cardId);
        cardElement.classList.add("selected");
    } else {
        selectedCards.splice(index, 1);
        cardElement.classList.remove("selected");
    }

    // Update play button state
    const playBtn = document.getElementById("play-btn");
    const clearBtn = document.getElementById("clear-selection-btn");
    playBtn.disabled = selectedCards.length === 0;
    clearBtn.disabled = selectedCards.length === 0;
}

function clearSelection() {
    selectedCards = [];
    document.querySelectorAll(".card.selected").forEach((card) => {
        card.classList.remove("selected");
    });
    document.getElementById("play-btn").disabled = true;
    document.getElementById("clear-selection-btn").disabled = true;
}

function updateTrick(trick) {
    const trickDisplay = document.getElementById("trick-display");
    if (!trick) {
        trickDisplay.innerHTML = '<p class="empty-trick">No trick in play - you can play anything</p>';
        return;
    }

    const cardsHtml = trick.cards
        .map((cardId) => {
            const rank = RANKS[Math.floor(cardId / 4)];
            const suit = SUITS[cardId % 4];
            const isRed = suit === "♥" || suit === "♦";
            const suitClass = isRed ? "suit-red" : "suit-black";
            return `
                <div class="card ${suitClass}">
                    <div class="card-rank ${suitClass}">${rank}</div>
                    <div class="card-suit ${suitClass}">${suit}</div>
                    <div class="card-rank-bottom ${suitClass}">${rank}</div>
                </div>
            `;
        })
        .join("");

    trickDisplay.innerHTML = `
        <div class="trick-info">
            <strong>${trick.display}</strong> (played by Player ${trick.player})
        </div>
        <div class="trick-cards">${cardsHtml}</div>
    `;
}

function updateOpponents(opponents) {
    const opponentsContainer = document.getElementById("opponents");
    opponentsContainer.innerHTML = "";

    opponents.forEach((opponent) => {
        const oppElement = document.createElement("div");
        oppElement.className = "opponent";
        if (opponent.is_current) {
            oppElement.classList.add("is-current");
        }
        oppElement.innerHTML = `
            <div class="opponent-name">${opponent.name}</div>
            <div class="opponent-cards">${opponent.cards_left} cards</div>
        `;
        opponentsContainer.appendChild(oppElement);
    });
}

function updateLegalMoves(moves) {
    const movesContainer = document.getElementById("legal-moves");
    movesContainer.innerHTML = "";

    if (moves.length === 0) {
        movesContainer.innerHTML = '<p style="color: #999;">No legal moves available</p>';
        return;
    }

    moves.forEach((move, index) => {
        const moveElement = document.createElement("div");
        moveElement.className = "move-option";
        moveElement.textContent = move.display;
        moveElement.addEventListener("click", () => playMoveByIndex(index));
        movesContainer.appendChild(moveElement);
    });
}

function updateBotSuggestion(botSuggestion, isHumanTurn) {
    const suggestionContainer = document.getElementById("bot-suggestion");

    if (!isHumanTurn) {
        suggestionContainer.innerHTML = '<p class="no-suggestion">Waiting for your turn...</p>';
        return;
    }

    if (!botSuggestion || !Array.isArray(botSuggestion) || botSuggestion.length === 0) {
        suggestionContainer.innerHTML = '<p class="no-suggestion">Unable to calculate bot suggestion</p>';
        return;
    }

    // Display the top moves with probabilities
    let suggestionHtml = `<div class="bot-suggestion-content">`;

    botSuggestion.forEach((move, index) => {
        const isTopMove = index === 0;
        const rankLabel = index === 0 ? "🥇" : index === 1 ? "🥈" : "🥉";

        suggestionHtml += `<div class="bot-move-item ${isTopMove ? "top-move" : ""}">`;
        suggestionHtml += `<div class="bot-move-header">`;
        suggestionHtml += `<span class="bot-move-rank">${rankLabel}</span>`;
        suggestionHtml += `<span class="bot-move-display">${move.display}</span>`;
        suggestionHtml += `<span class="bot-move-probability">${move.probability_pct}%</span>`;
        suggestionHtml += `</div>`;

        // Probability bar
        suggestionHtml += `<div class="probability-bar-container">`;
        suggestionHtml += `<div class="probability-bar" style="width: ${move.probability_pct}%"></div>`;
        suggestionHtml += `</div>`;

        // If the move has cards, show them visually (only for top move to save space)
        if (isTopMove && move.cards && move.cards.length > 0) {
            const cardsHtml = move.cards
                .map((cardId) => {
                    const rank = RANKS[Math.floor(cardId / 4)];
                    const suit = SUITS[cardId % 4];
                    const isRed = suit === "♥" || suit === "♦";
                    const suitClass = isRed ? "suit-red" : "suit-black";
                    return `
                        <div class="card small-card ${suitClass}">
                            <div class="card-rank ${suitClass}">${rank}</div>
                            <div class="card-suit ${suitClass}">${suit}</div>
                            <div class="card-rank-bottom ${suitClass}">${rank}</div>
                        </div>
                    `;
                })
                .join("");
            suggestionHtml += `<div class="bot-suggestion-cards">${cardsHtml}</div>`;
        }

        suggestionHtml += `</div>`;
    });

    suggestionHtml += `</div>`;
    suggestionContainer.innerHTML = suggestionHtml;
}

async function playMoveByIndex(moveIndex) {
    if (!currentGameId) return;

    const playBtn = document.getElementById("play-btn");
    playBtn.disabled = true;
    playBtn.innerHTML = '<span class="loading"></span> Playing...';

    try {
        const response = await fetch(`/api/game/${currentGameId}/action`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                move_index: moveIndex,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to play move");
        }

        const result = await response.json();
        selectedCards = [];
        updateGameDisplay(result);

        // Show AI actions if any
        if (result.ai_actions && result.ai_actions.length > 0) {
            showAIActions(result.ai_actions);
        }
    } catch (error) {
        alert(`Error playing move: ${error.message}`);
        console.error(error);
    } finally {
        playBtn.disabled = false;
        playBtn.innerHTML = "Play Selected";
    }
}

async function playSelectedCards() {
    if (!currentGameId || selectedCards.length === 0) return;

    const playBtn = document.getElementById("play-btn");
    playBtn.disabled = true;
    playBtn.innerHTML = '<span class="loading"></span> Playing...';

    try {
        const response = await fetch(`/api/game/${currentGameId}/action`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                cards: selectedCards,
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || "Failed to play cards");
        }

        const result = await response.json();
        selectedCards = [];
        clearSelection();
        updateGameDisplay(result);

        // Show AI actions if any
        if (result.ai_actions && result.ai_actions.length > 0) {
            showAIActions(result.ai_actions);
        }
    } catch (error) {
        alert(`Error playing cards: ${error.message}`);
        console.error(error);
    } finally {
        playBtn.disabled = false;
        playBtn.innerHTML = "Play Selected";
    }
}

async function playPass() {
    if (!currentGameId) return;

    // Find PASS move index
    const passIndex = legalMoves.findIndex((move) => move.type === 0); // PASS type is 0
    if (passIndex === -1) {
        alert("Pass is not a legal move right now");
        return;
    }

    await playMoveByIndex(passIndex);
}

function showAIActions(aiActions) {
    const logContainer = document.getElementById("ai-actions-log");
    logContainer.innerHTML = "<h4>AI Actions:</h4>";

    aiActions.forEach((action) => {
        const actionElement = document.createElement("div");
        actionElement.className = "ai-action-item";
        actionElement.innerHTML = `<strong>Player ${action.player}:</strong> ${action.action.display}`;
        logContainer.appendChild(actionElement);
    });

    // Auto-scroll to bottom
    logContainer.scrollTop = logContainer.scrollHeight;
}

function showGameOver(gameState) {
    document.getElementById("game-area").classList.add("hidden");
    document.getElementById("game-over").classList.remove("hidden");

    const title = document.getElementById("game-over-title");
    const message = document.getElementById("game-over-message");

    if (gameState.winner === gameState.human_player) {
        title.textContent = "🎉 Congratulations!";
        message.textContent = "You won the game!";
    } else {
        title.textContent = "Game Over";
        message.textContent = `Player ${gameState.winner} won the game. Better luck next time!`;
    }
}

