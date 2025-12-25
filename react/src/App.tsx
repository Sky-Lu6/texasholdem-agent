import { useState, useEffect } from 'react'
import './App.css'
import Analytics from './Analytics'

const API_URL = 'http://localhost:8000'

interface Player {
  id: number
  name: string
  state: string
  chips: number
  in_pot: number
  current_bet: number
  hand: string[]
  visible: boolean
}

interface RaiseRange {
  min: number
  max: number
}

interface AvailableActions {
  CALL: boolean
  CHECK: boolean
  FOLD: boolean
  RAISE: boolean
  raiseRange?: RaiseRange
}

interface PotView {
  id: number
  amount: number
  eligible_players: number[]
}

interface GameState {
  players: Player[]
  board: string[]
  history: string[]
  availableActions: AvailableActions
  isHandRunning: boolean
  isGameRunning: boolean
  currentPlayer: number
  totalPot: number
  pots: PotView[]
}

interface GameMode {
  mode: string
  maxPlayers: number
  heroId: number | null
  buyin: number
  bigBlind: number
  smallBlind: number
}

// Component to display pots (main + side pots)
function PotsDisplay({ pots, totalPot }: { pots: PotView[], totalPot: number }) {
  if (!pots || pots.length === 0) {
    return <div className="pot-info">💰 Pot: {totalPot}</div>
  }

  if (pots.length === 1) {
    // Single pot - simple display
    return <div className="pot-info">💰 Pot: {pots[0].amount}</div>
  }

  // Multiple pots - show main + side pots
  return (
    <div className="pots-display">
      <div className="total-pot">💰 Total: {totalPot}</div>
      <div className="pot-breakdown">
        {pots.map((pot, index) => (
          <div key={pot.id} className={`pot-item ${index === 0 ? 'main-pot' : 'side-pot'}`}>
            <span className="pot-label">
              {index === 0 ? 'Main Pot' : `Side Pot ${index}`}
            </span>
            <span className="pot-amount">${pot.amount}</span>
            <span className="pot-players">
              ({pot.eligible_players.length} {pot.eligible_players.length === 1 ? 'player' : 'players'})
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

function App() {
  const [gameState, setGameState] = useState<GameState | null>(null)
  const [gameMode, setGameMode] = useState<GameMode | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [raiseAmount, setRaiseAmount] = useState<string>('')
  const [showModeSelector, setShowModeSelector] = useState(false)
  const [showOpponentCards, setShowOpponentCards] = useState(false)
  const [showAnalytics, setShowAnalytics] = useState(false)

  // Frontend playback controls
  const [speed, setSpeed] = useState<number>(5) // 1 slow – 10 fast
  const [visibleHistoryCount, setVisibleHistoryCount] = useState(0)
  const [currentActorId, setCurrentActorId] = useState<number | null>(null)
  const [currentActionText, setCurrentActionText] = useState<string>('')
  const [renderTick, setRenderTick] = useState(0)

  const API_TIMEOUT = 30000  // 30 seconds for multi-player games

  const speedToDelay = (s: number) => {
    const clamped = Math.min(10, Math.max(1, s))
    // 1 -> 1300ms, 10 -> 100ms
    return 1300 - clamped * 120
  }

  const delay = speedToDelay(speed)

  const sleep = (ms: number) => new Promise(res => setTimeout(res, ms))

  const fetchState = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await fetch(`${API_URL}/state`, {
        signal: AbortSignal.timeout(API_TIMEOUT)
      })
      if (!response.ok) throw new Error('Failed to fetch game state')
      const data = await response.json()
      setGameState(data)
      setRenderTick(t => t + 1)
    } catch (err) {
      if (err instanceof Error && err.name !== 'TimeoutError') {
        setError(err.message)
      } else if (err instanceof Error && err.name === 'TimeoutError') {
        console.warn('State fetch timeout')
      } else {
        setError('Unknown error')
      }
    } finally {
      setLoading(false)
    }
  }

  const fetchGameMode = async () => {
    try {
      const response = await fetch(`${API_URL}/game-mode`)
      if (!response.ok) throw new Error('Failed to fetch game mode')
      const data = await response.json()
      setGameMode(data)
    } catch (err) {
      console.error('Failed to fetch game mode:', err)
    }
  }

  const takeAction = async (actionType: string, total?: number) => {
    try {
      setLoading(true)
      setError(null)
      const response = await fetch(`${API_URL}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type: actionType, total })
      })
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Action failed')
      }
      const data = await response.json()
      // Small pause for smoother transition
      await sleep(120)
      setGameState(data)
      setRaiseAmount('')
      setRenderTick(t => t + 1)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const newHand = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await fetch(`${API_URL}/new-hand`, { method: 'POST' })
      if (!response.ok) throw new Error('Failed to start new hand')
      const data = await response.json()
      setGameState(data)
      setRenderTick(t => t + 1)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const setMode = async (mode: string, maxPlayers: number) => {
    try {
      setLoading(true)
      setError(null)
      const response = await fetch(
        `${API_URL}/set-mode?mode=${mode}&max_players=${maxPlayers}`,
        { method: 'POST' }
      )
      if (!response.ok) throw new Error('Failed to set mode')
      await fetchGameMode()
      await fetchState()
      setShowModeSelector(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const resetGame = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await fetch(`${API_URL}/reset-game`, { method: 'POST' })
      if (!response.ok) throw new Error('Failed to reset game')
      const data = await response.json()
      setGameState(data)
      setRenderTick(t => t + 1)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  // Initial load
  useEffect(() => {
    fetchState()
    fetchGameMode()
  }, [])

  // Background refresh (same logic as before; this just keeps state fresh)
  useEffect(() => {
    const interval = setInterval(() => {
      if (!gameState || gameState.currentPlayer !== 0 || !gameState.isHandRunning) {
        fetchState()
      }
    }, 2000)
    return () => clearInterval(interval)
  }, [gameState?.currentPlayer, gameState?.isHandRunning])

  // Reset history replay when a new hand or history changes
  useEffect(() => {
    if (!gameState) return
    setVisibleHistoryCount(0)
    setCurrentActorId(null)
    setCurrentActionText('')
  }, [gameState?.history, gameState?.isHandRunning, gameState?.totalPot])

  // Step through history on the client side
  useEffect(() => {
    if (!gameState) return
    const lines = gameState.history
    if (!lines.length) return
    if (visibleHistoryCount >= lines.length) return

    const id = setTimeout(() => {
      setVisibleHistoryCount(c => Math.min(c + 1, lines.length))
    }, delay)

    return () => clearTimeout(id)
  }, [gameState?.history, visibleHistoryCount, delay])

  // Derive current actor/action text from the last visible history line
  useEffect(() => {
    if (!gameState) return
    const lines = gameState.history
    if (!lines.length || visibleHistoryCount === 0) {
      setCurrentActorId(null)
      setCurrentActionText('')
      return
    }

    const lastLine = lines[Math.min(visibleHistoryCount - 1, lines.length - 1)]

    // Adjust this parsing to match your real history format
    // Example expected format: "Player 2: CALL 50"
    const match = lastLine.match(/Player\s+(\d+):\s*(.+)$/i)
    if (match) {
      const id = parseInt(match[1], 10)
      const actionText = match[2]
      setCurrentActorId(id)
      setCurrentActionText(actionText)
    } else {
      setCurrentActorId(null)
      setCurrentActionText(lastLine)
    }
  }, [gameState?.history, visibleHistoryCount])

  if (loading && !gameState) {
    return (
      <div className="centered-container">
        <h2>Loading...</h2>
      </div>
    )
  }

  if (error && !gameState) {
    return (
      <div className="centered-container">
        <h2>Error: {error}</h2>
        <button onClick={fetchState}>Retry</button>
      </div>
    )
  }

  if (!gameState) return null

  const isSpectatorMode = gameMode?.mode === 'spectator'
  const heroId = gameMode?.heroId ?? 0
  const myPlayer = gameState.players.find(p => p.id === heroId)
  const isMyTurn =
    !isSpectatorMode && gameState.currentPlayer === heroId && gameState.isHandRunning
  const actions = gameState.availableActions

  const gameOver =
    gameState.players.filter(p => p.chips > 0).length <= 1 && !gameState.isHandRunning

  // Players layout (clockwise) for up to 5 opponents
  const opponents = [...gameState.players]
    .filter(p => p.id !== heroId)
    .sort((a, b) => a.id - b.id)

  const playerPositions: Record<number, string> = {}
  let opponentsAbove: Player[] = []
  let opponentsLeft: Player[] = []
  let opponentsRight: Player[] = []

  if (opponents.length === 1) {
    opponentsAbove = [opponents[0]]
    playerPositions[opponents[0].id] = 'top'
  } else if (opponents.length === 2) {
    opponentsAbove = opponents
    playerPositions[opponents[0].id] = 'top-left'
    playerPositions[opponents[1].id] = 'top-right'
  } else if (opponents.length === 3) {
    opponentsAbove = [opponents[0], opponents[1], opponents[2]]
    playerPositions[opponents[0].id] = 'top-left'
    playerPositions[opponents[1].id] = 'top'
    playerPositions[opponents[2].id] = 'top-right'
  } else if (opponents.length === 5) {
    // 1 left, 2-4 top (L→R), 5 right
    opponentsLeft = [opponents[0]]
    opponentsAbove = [opponents[1], opponents[2], opponents[3]]
    opponentsRight = [opponents[4]]

    playerPositions[opponents[0].id] = 'left'
    playerPositions[opponents[1].id] = 'top-left'
    playerPositions[opponents[2].id] = 'top'
    playerPositions[opponents[3].id] = 'top-right'
    playerPositions[opponents[4].id] = 'right'
  } else {
    // Fallback: just put everyone above
    opponentsAbove = opponents
  }

  const heroPlayer = gameState.players.find(p => p.id === heroId)
  if (heroPlayer) {
    playerPositions[heroPlayer.id] = 'bottom'
  }

  const renderOpponent = (player: Player) => {
    const isCurrentPlayer = player.id === gameState.currentPlayer
    const isLastActor = player.id === currentActorId
    return (
      <div
        key={player.id}
        className={
          'player-section opponent ' +
          (isCurrentPlayer ? 'active ' : '') +
          (isLastActor ? 'last-actor ' : '')
        }
      >
        <div className="player-info">
          <h3>
            🤖 {player.name}
            {isCurrentPlayer && ' ⏰'}
          </h3>
          <div className="chips">💰 {player.chips}</div>
          <div className="in-pot">Pot: {player.in_pot}</div>
        </div>
        <div className="hand">
          {showOpponentCards && player.hand && player.hand.length > 0 ? (
            player.hand.map((card: string, i: number) => (
              <span key={i} className="card">
                {card}
              </span>
            ))
          ) : (
            <>
              <span className="card back">🂠</span>
              <span className="card back">🂠</span>
            </>
          )}
        </div>
      </div>
    )
  }

  // Show analytics view if toggled
  if (showAnalytics) {
    return (
      <div className="analytics-wrapper">
        <div className="analytics-nav">
          <button
            className="btn btn-back"
            onClick={() => setShowAnalytics(false)}
          >
            ← Back to Game
          </button>
        </div>
        <Analytics />
      </div>
    )
  }

  return (
    <div className="poker-game">
      {/* LEFT: header + table + hero */}
      <div className="poker-left">
        <div className="header">
          <h1>
            🃏 Texas Hold&apos;em {isSpectatorMode ? 'AI Battle' : 'vs DQN Agent'}
          </h1>
          <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
            <button
              className="btn btn-analytics"
              onClick={() => setShowAnalytics(true)}
            >
              📊 Analytics
            </button>
            <button
              className="btn btn-mode"
              onClick={() => setShowModeSelector(!showModeSelector)}
              disabled={loading}
            >
              ⚙️ {isSpectatorMode ? 'Spectator' : 'Player'} Mode
            </button>
            <button
              className="btn btn-mode"
              onClick={() => setShowOpponentCards(!showOpponentCards)}
              disabled={loading}
            >
              👁️ {showOpponentCards ? 'Hide' : 'Show'} Opponent Cards
            </button>
            <button
              className="btn btn-reset"
              onClick={resetGame}
              disabled={loading}
            >
              🔄 Reset
            </button>

            {/* Speed slider 1–10 */}
            <div className="speed-control">
              <label htmlFor="speed">Speed</label>
              <input
                id="speed"
                type="range"
                min={1}
                max={10}
                step={1}
                value={speed}
                onChange={(e) => setSpeed(parseInt(e.target.value))}
              />
              <span>{speed}</span>
            </div>
          </div>
        </div>

        {showModeSelector && (
          <div className="mode-selector">
            <h3>Select Game Mode</h3>
            <div className="mode-options">
              <div className="mode-option">
                <h4>👤 Player Mode</h4>
                <p>Play against AI opponents</p>
                <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                  <button className="btn btn-select" onClick={() => setMode('player', 2)}>
                    1 vs 1 AI
                  </button>
                  <button className="btn btn-select" onClick={() => setMode('player', 3)}>
                    1 vs 2 AI
                  </button>
                  <button className="btn btn-select" onClick={() => setMode('player', 4)}>
                    1 vs 3 AI
                  </button>
                  <button className="btn btn-select" onClick={() => setMode('player', 6)}>
                    1 vs 5 AI
                  </button>
                </div>
              </div>
              <div className="mode-option">
                <h4>👁️ Spectator Mode</h4>
                <p>Watch AI agents battle</p>
                <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                  <button className="btn btn-select" onClick={() => setMode('spectator', 2)}>
                    2 Players
                  </button>
                  <button className="btn btn-select" onClick={() => setMode('spectator', 3)}>
                    3 Players
                  </button>
                  <button className="btn btn-select" onClick={() => setMode('spectator', 4)}>
                    4 Players
                  </button>
                  <button className="btn btn-select" onClick={() => setMode('spectator', 6)}>
                    6 Players
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {error && <div className="error-banner">{error}</div>}

        {gameOver && (
          <div className="game-over-banner">
            <h2>🎮 Game Over!</h2>
            <p>
              {isSpectatorMode
                ? `Winner: Player ${
                    gameState.players.find(p => p.chips > 0)?.id
                  }!`
                : myPlayer?.chips === 0
                ? '💀 You lost!'
                : myPlayer && myPlayer.chips > 0 && gameState.players.filter(p => p.chips > 0).length === 1
                ? '🎉 You won!'
                : 'Game Over'}
            </p>
            <button
              className="btn btn-new-game"
              onClick={resetGame}
              disabled={loading}
            >
              Start New Game
            </button>
          </div>
        )}

        {/* Table + hero section */}
        <div className={`table-and-hero render-${renderTick % 2}`}>
          {/* Current action banner like PokerTH */}
          {currentActionText && (
            <div className="current-action-banner">
              {currentActorId !== null && (
                <span className="current-action-player">Player {currentActorId}</span>
              )}
              <span className="current-action-text">{currentActionText}</span>
            </div>
          )}

          <div className="poker-table-container">
            {opponentsAbove.length > 0 && (
              <div className="opponents-above">
                {opponentsAbove.map(renderOpponent)}
              </div>
            )}

            <div className="table-area">
              {opponentsLeft.length > 0 && (
                <div className="opponents-left">
                  {opponentsLeft.map(renderOpponent)}
                </div>
              )}

              <div className="poker-table">
                {/* Chip stacks by position */}
                {gameState.players.map(player => {
                  const pos = playerPositions[player.id]
                  if (!pos || player.current_bet <= 0) return null
                  return (
                    <div
                      key={`chip-${player.id}`}
                      className={`chip-stack chip-stack-${pos}`}
                    >
                      💰 {player.current_bet}
                    </div>
                  )
                })}

                <div className="board-section">
                  <PotsDisplay pots={gameState.pots} totalPot={gameState.totalPot} />
                  <h3>Community Cards</h3>
                  <div className="board">
                    {gameState.board.length ? (
                      gameState.board.map((card, i) => (
                        <span key={i} className="card">
                          {card}
                        </span>
                      ))
                    ) : (
                      <span className="empty">No cards yet</span>
                    )}
                  </div>
                </div>
              </div>

              {opponentsRight.length > 0 && (
                <div className="opponents-right">
                  {opponentsRight.map(renderOpponent)}
                </div>
              )}
            </div>

            {heroPlayer && (
              <div
                className={
                  'player-section player hero-position ' +
                  (heroPlayer.id === gameState.currentPlayer ? 'active ' : '')
                }
              >
                <div className="player-info">
                  <h3>
                    👤 {heroPlayer.name} (You)
                    {heroPlayer.id === gameState.currentPlayer && ' ⏰'}
                  </h3>
                  <div className="chips">💰 Chips: {heroPlayer.chips}</div>
                  <div className="in-pot">In Pot: {heroPlayer.in_pot}</div>
                </div>
                <div className="hand">
                  {heroPlayer.hand.length ? (
                    heroPlayer.hand.map((card: string, i: number) => (
                      <span key={i} className="card">
                        {card}
                      </span>
                    ))
                  ) : (
                    <>
                      <span className="card back">🂠</span>
                      <span className="card back">🂠</span>
                    </>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* RIGHT: actions + replaying history */}
      <div className="poker-right">
        {!isSpectatorMode && (
          <div className="actions-section">
            {gameState.isHandRunning ? (
              isMyTurn ? (
                <div className="action-buttons">
                  <h3>Your Turn - Choose Action:</h3>

                  <div className="buttons">
                    {actions.FOLD && (
                      <button
                        className="btn btn-fold"
                        onClick={() => takeAction('FOLD')}
                        disabled={loading}
                      >
                        Fold
                      </button>
                    )}
                    {actions.CHECK && (
                      <button
                        className="btn btn-check"
                        onClick={() => takeAction('CHECK')}
                        disabled={loading}
                      >
                        Check
                      </button>
                    )}
                    {actions.CALL && (
                      <button
                        className="btn btn-call"
                        onClick={() => takeAction('CALL')}
                        disabled={loading}
                      >
                        Call
                      </button>
                    )}
                  </div>

                  {actions.RAISE && actions.raiseRange && (
                    <>
                      <div className="quick-raise-buttons">
                        <button
                          className="btn btn-quick-raise"
                          onClick={() => {
                            const amount = Math.min(
                              Math.max(
                                Math.floor(gameState.totalPot * 0.5),
                                actions.raiseRange!.min
                              ),
                              actions.raiseRange!.max
                            )
                            takeAction('RAISE', amount)
                          }}
                          disabled={loading}
                        >
                          1/2 Pot
                        </button>
                        <button
                          className="btn btn-quick-raise"
                          onClick={() => {
                            const amount = Math.min(
                              Math.max(
                                gameState.totalPot,
                                actions.raiseRange!.min
                              ),
                              actions.raiseRange!.max
                            )
                            takeAction('RAISE', amount)
                          }}
                          disabled={loading}
                        >
                          Pot
                        </button>
                        <button
                          className="btn btn-quick-raise"
                          onClick={() => {
                            const amount = Math.min(
                              Math.max(
                                Math.floor(gameState.totalPot * 2),
                                actions.raiseRange!.min
                              ),
                              actions.raiseRange!.max
                            )
                            takeAction('RAISE', amount)
                          }}
                          disabled={loading}
                        >
                          2x Pot
                        </button>
                        <button
                          className="btn btn-quick-raise"
                          onClick={() =>
                            takeAction('RAISE', actions.raiseRange!.max)
                          }
                          disabled={loading}
                        >
                          All-In
                        </button>
                      </div>
                      <div className="raise-section">
                        <input
                          type="number"
                          placeholder={`Custom: ${actions.raiseRange.min}-${actions.raiseRange.max}`}
                          value={raiseAmount}
                          onChange={e => setRaiseAmount(e.target.value)}
                          min={actions.raiseRange.min}
                          max={actions.raiseRange.max}
                          disabled={loading}
                        />
                        <button
                          className="btn btn-raise"
                          onClick={() => {
                            const amount = parseInt(raiseAmount)
                            if (
                              amount >= actions.raiseRange!.min &&
                              amount <= actions.raiseRange!.max
                            ) {
                              takeAction('RAISE', amount)
                            } else {
                              setError(
                                `Raise amount must be between ${actions.raiseRange!.min} and ${actions.raiseRange!.max}`
                              )
                            }
                          }}
                          disabled={loading || !raiseAmount}
                        >
                          Raise to {raiseAmount || '...'}
                        </button>
                      </div>
                    </>
                  )}
                </div>
              ) : (
                <div className="waiting">
                  <h3>⏳ Waiting for DQN Agent...</h3>
                </div>
              )
            ) : (
              <div className="hand-over">
                <h3>Hand Over</h3>
                <button
                  className="btn btn-new-hand"
                  onClick={newHand}
                  disabled={loading}
                >
                  Start New Hand
                </button>
              </div>
            )}
          </div>
        )}

        {/* Action history replayed on the right */}
        <div className="history-section">
          <h3>Action History</h3>
          <div className="history">
            {gameState.history.length ? (
              gameState.history
                .slice(0, visibleHistoryCount)
                .map((line, i) => (
                  <div key={i} className="history-line">
                    {line}
                  </div>
                ))
            ) : (
              <div className="empty">No actions yet</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
