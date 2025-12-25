import { useState, useEffect } from 'react'
import './Analytics.css'

const API_URL = 'http://localhost:8000'

interface PlayerStats {
  hands_played: number
  hands_won: number
  total_invested: number
  total_won: number
  win_rate: number
  avg_pot_size: number
  actions: Record<string, number>
}

interface AnalyticsData {
  total_hands: number
  player_stats: Record<string, PlayerStats>
  action_frequency: Record<string, number>
  phase_distribution: Record<string, number>
  pot_size_distribution: number[]
  winner_distribution: Record<string, number>
  avg_pot_size: number
}

function Analytics() {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchAnalytics = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await fetch(`${API_URL}/analytics/hand-history`)
      if (!response.ok) {
        throw new Error('Failed to fetch analytics')
      }
      const data = await response.json()
      setAnalytics(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchAnalytics()
  }, [])

  if (loading) {
    return (
      <div className="analytics-container">
        <div className="loading">Loading analytics...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="analytics-container">
        <div className="error">Error: {error}</div>
        <button onClick={fetchAnalytics}>Retry</button>
      </div>
    )
  }

  if (!analytics || analytics.total_hands === 0) {
    return (
      <div className="analytics-container">
        <h1>Hand History Analytics</h1>
        <div className="no-data">
          <p>No hand history data available yet.</p>
          <p>Play some hands and export them to generate analytics!</p>
        </div>
      </div>
    )
  }

  // Calculate some derived stats
  const playerIds = Object.keys(analytics.player_stats)
  const totalActions = Object.values(analytics.action_frequency).reduce((a, b) => a + b, 0)

  return (
    <div className="analytics-container">
      <div className="analytics-header">
        <h1>📊 Hand History Analytics</h1>
        <button onClick={fetchAnalytics} className="refresh-btn">
          🔄 Refresh
        </button>
      </div>

      {/* Summary Cards */}
      <div className="summary-cards">
        <div className="stat-card">
          <div className="stat-label">Total Hands</div>
          <div className="stat-value">{analytics.total_hands}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg Pot Size</div>
          <div className="stat-value">{analytics.avg_pot_size.toFixed(0)}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Total Actions</div>
          <div className="stat-value">{totalActions}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Players</div>
          <div className="stat-value">{playerIds.length}</div>
        </div>
      </div>

      {/* Player Statistics */}
      <div className="section">
        <h2>Player Statistics</h2>
        <div className="player-stats-grid">
          {playerIds.map(playerId => {
            const stats = analytics.player_stats[playerId]
            return (
              <div key={playerId} className="player-stat-card">
                <h3>Player {playerId}</h3>
                <div className="player-stats">
                  <div className="stat-row">
                    <span>Hands Played:</span>
                    <strong>{stats.hands_played}</strong>
                  </div>
                  <div className="stat-row">
                    <span>Hands Won:</span>
                    <strong>{stats.hands_won}</strong>
                  </div>
                  <div className="stat-row">
                    <span>Win Rate:</span>
                    <strong className={stats.win_rate > 50 ? 'positive' : 'negative'}>
                      {stats.win_rate.toFixed(1)}%
                    </strong>
                  </div>
                  <div className="stat-row">
                    <span>Total Won:</span>
                    <strong className={stats.total_won > stats.total_invested ? 'positive' : 'negative'}>
                      {stats.total_won.toFixed(0)} chips
                    </strong>
                  </div>
                  <div className="stat-row">
                    <span>Net Profit:</span>
                    <strong className={(stats.total_won - stats.total_invested) > 0 ? 'positive' : 'negative'}>
                      {(stats.total_won - stats.total_invested).toFixed(0)} chips
                    </strong>
                  </div>
                </div>

                {/* Player Actions */}
                {Object.keys(stats.actions).length > 0 && (
                  <div className="action-breakdown">
                    <h4>Actions</h4>
                    {Object.entries(stats.actions).map(([action, count]) => (
                      <div key={action} className="action-bar">
                        <span className="action-name">{action}</span>
                        <div className="action-progress">
                          <div
                            className="action-fill"
                            style={{
                              width: `${(count / stats.hands_played) * 100}%`
                            }}
                          />
                        </div>
                        <span className="action-count">{count}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Action Distribution */}
      <div className="section">
        <h2>Action Distribution</h2>
        <div className="chart">
          {Object.entries(analytics.action_frequency).map(([action, count]) => {
            const percentage = (count / totalActions) * 100
            return (
              <div key={action} className="bar-item">
                <div className="bar-label">{action}</div>
                <div className="bar-container">
                  <div
                    className="bar-fill"
                    style={{ width: `${percentage}%` }}
                  >
                    <span className="bar-text">{count} ({percentage.toFixed(1)}%)</span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Phase Distribution */}
      {Object.keys(analytics.phase_distribution).length > 0 && (
        <div className="section">
          <h2>Hand Phase Distribution</h2>
          <div className="chart">
            {Object.entries(analytics.phase_distribution).map(([phase, count]) => {
              const percentage = (count / analytics.total_hands) * 100
              return (
                <div key={phase} className="bar-item">
                  <div className="bar-label">{phase}</div>
                  <div className="bar-container">
                    <div
                      className="bar-fill phase-fill"
                      style={{ width: `${percentage}%` }}
                    >
                      <span className="bar-text">{count} ({percentage.toFixed(1)}%)</span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Pot Size Distribution */}
      {analytics.pot_size_distribution.length > 0 && (
        <div className="section">
          <h2>Pot Size Distribution</h2>
          <div className="pot-sizes">
            <div className="stat-row">
              <span>Min Pot:</span>
              <strong>{Math.min(...analytics.pot_size_distribution)}</strong>
            </div>
            <div className="stat-row">
              <span>Max Pot:</span>
              <strong>{Math.max(...analytics.pot_size_distribution)}</strong>
            </div>
            <div className="stat-row">
              <span>Average:</span>
              <strong>{analytics.avg_pot_size.toFixed(0)}</strong>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Analytics
