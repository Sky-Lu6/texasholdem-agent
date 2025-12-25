# 🃏 AI-Powered Texas Hold'em Poker

A full-stack web application featuring an intelligent poker AI trained using Deep Q-Network (DQN) reinforcement learning. Play against a neural network that has learned optimal poker strategy through thousands of training hands.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.121-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.2-blue.svg)](https://reactjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue.svg)](https://www.typescriptlang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Powered-red.svg)](https://pytorch.org/)

## 🎯 Project Overview

This project combines **reinforcement learning**, **full-stack web development**, and **game theory** to create an intelligent poker opponent. The AI agent is trained using Deep Q-Learning to make strategic decisions including when to bet, raise, call, or fold based on hand strength, pot odds, and game state.

### Key Features

- 🤖 **DQN-Trained AI Agent**: Neural network trained on 1000+ hands using Deep Q-Learning
- 🎮 **Interactive Web Interface**: Smooth, animated React frontend with real-time game updates
- 📊 **Hand History Analytics**: Comprehensive statistics and gameplay analysis
- 🏆 **Multiple Game Modes**: Play against AI or watch AI vs AI battles
- 💰 **Side Pot Management**: Proper handling of all-in scenarios with multiple players
- 📁 **PGN Export**: Save and replay game history in standard poker notation
- ⚡ **Real-time State Sync**: FastAPI backend with WebSocket-ready architecture

## 🏗️ Architecture

```
┌─────────────────┐
│   React + TS    │  ← Interactive poker table UI with animations
│   Frontend      │
└────────┬────────┘
         │ REST API
         ↓
┌─────────────────┐
│   FastAPI       │  ← Game state management & API endpoints
│   Backend       │
└────────┬────────┘
         │
    ┌────┴─────┐
    ↓          ↓
┌────────┐  ┌──────────────┐
│  Game  │  │   DQN Agent  │  ← Neural network decision-making
│ Engine │  │   (PyTorch)  │
└────────┘  └──────────────┘
```

## 🧠 AI Architecture

The DQN agent uses a 4-layer fully-connected neural network:

**Input (116 features)**:
- Hero's hole cards (52-dim binary vector)
- Board cards (52-dim binary vector)
- Hand phase encoding (4-dim one-hot: preflop/flop/turn/river)
- Poker features (8-dim): pot size, chips to call, stack sizes, pot odds, position, etc.

**Network Architecture**:
```
Input (116) → FC(128) → ReLU → FC(128) → ReLU → FC(64) → ReLU → FC(8)
```

**Output (8 actions)**:
1. FOLD
2. CHECK
3. CALL
4. RAISE_MIN (minimum legal raise)
5. RAISE_HALF_POT (0.5x pot)
6. RAISE_POT (1x pot)
7. RAISE_2X_POT (2x pot)
8. ALL_IN

**Training Details**:
- Algorithm: Deep Q-Learning with experience replay
- Replay buffer: 50,000 transitions
- Epsilon-greedy exploration: 90% → 5% (decay: 0.9995)
- Reward shaping: Win probability tracking + outcome-based rewards
- Target network: Soft updates (τ = 0.005)
- Optimizer: Adam (lr = 0.001)

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- pip and npm

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Sky-Lu6/texasholdem-agent.git
cd texasholdem-agent
```

2. **Set up the backend**
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Set up the frontend**
```bash
cd react
npm install
```

### Running the Application

1. **Start the backend server**
```bash
# From project root, with venv activated
python api_server.py
```
Backend will run at `http://localhost:8000`

2. **Start the frontend (in a new terminal)**
```bash
cd react
npm run dev
```
Frontend will run at `http://localhost:5173`

3. **Open your browser** and navigate to `http://localhost:5173`

## 🎮 How to Play

1. **Start a New Game**: The game automatically deals a hand when you load the page
2. **Make Your Move**: Use the action buttons to FOLD, CHECK, CALL, or RAISE
3. **Watch the AI**: The DQN agent will make strategic decisions based on its training
4. **View Analytics**: Click "Show Analytics" to see detailed statistics from your game history
5. **Switch Modes**: Toggle between playing against the AI or watching AI vs AI

### Game Controls

- **FOLD**: Give up your hand and forfeit the pot
- **CHECK**: Pass without betting (only available when there's no bet to call)
- **CALL**: Match the current bet
- **RAISE**: Increase the bet (use slider to select amount)
- **Show/Hide Opponent Cards**: Toggle opponent card visibility (for learning)

## 📊 Analytics Dashboard

The analytics page provides insights into gameplay:
- **Total hands played**: Overall game statistics
- **Player win rates**: Performance breakdown per player
- **Action frequency**: Distribution of actions (fold/call/raise)
- **Average pot sizes**: Betting patterns analysis

## 🧪 Training Your Own AI

To train the DQN agent from scratch:

```bash
# Quick training (100 episodes)
python train_fast.py

# Extended training (1000+ episodes)
python texasholdem/rl/train_dqn.py
```

Training checkpoints are saved to `./checkpoints/`:
- `holdem_dqn_best.pt`: Best performing model
- `holdem_dqn.pt`: Most recent model
- `replay.pkl`: Experience replay buffer
- `train_state.pkl`: Training progress (epsilon, win rate, etc.)

### Training Metrics

Monitor training with:
- Episode reward (chip profit/loss)
- 100-episode moving average
- Win rate percentage
- Training loss

## 📁 Project Structure

```
texasholdem/
├── api_server.py              # FastAPI backend server
├── react/                     # React frontend
│   ├── src/
│   │   ├── App.tsx           # Main poker game UI
│   │   ├── Analytics.tsx     # Statistics dashboard
│   │   └── App.css           # Styles and animations
│   └── package.json
├── texasholdem/              # Core poker engine
│   ├── game/                 # Game logic and rules
│   ├── evaluator/            # Hand strength evaluation
│   ├── card/                 # Card representations
│   └── rl/                   # Reinforcement learning
│       ├── env.py            # RL environment
│       ├── dqn_agent.py      # DQN agent (inference)
│       ├── train_dqn.py      # DQN training script
│       └── win_probability.py # Monte Carlo equity calculation
├── checkpoints/              # Trained model weights
├── pgns/                     # Game history files
└── docs/                     # Documentation
```

## 🔧 Technical Highlights

### Backend (Python)
- **FastAPI**: Modern, fast API framework with automatic OpenAPI docs
- **PyTorch**: Deep learning framework for DQN implementation
- **Pydantic**: Data validation and serialization
- **NumPy**: Efficient array operations for game state representation

### Frontend (TypeScript/React)
- **React 19**: Modern UI with hooks and functional components
- **TypeScript**: Type-safe frontend development
- **Vite**: Fast build tool and dev server
- **CSS Animations**: Smooth card dealing and chip movements

### AI/ML
- **Deep Q-Learning**: Value-based RL with experience replay
- **Reward Shaping**: Sophisticated reward function encouraging strategic play
- **Monte Carlo Simulation**: Real-time win probability estimation
- **Legal Action Masking**: Ensures only valid poker moves

## 📈 Performance Metrics

- **AI Win Rate**: 65%+ against random baseline
- **Training Time**: ~2 hours for 1000 episodes (CPU)
- **Inference Speed**: <50ms per decision
- **API Response Time**: <100ms average

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- Reinforcement learning (Q-learning, experience replay, reward shaping)
- Full-stack development (REST APIs, React, TypeScript)
- Neural network architecture design (PyTorch)
- Game theory and poker strategy
- State management and real-time updates
- Code organization and software architecture

## 📝 Future Enhancements

- [ ] Deploy to cloud platform (Vercel + Render)
- [ ] Add multiplayer support (3-6 players)
- [ ] Implement Deep CFR algorithm for comparison
- [ ] Add web sockets for real-time spectator mode
- [ ] Create mobile-responsive design
- [ ] Add difficulty levels (beginner/intermediate/expert AI)
- [ ] Implement tournament mode
- [ ] Add sound effects and enhanced animations

## 🤝 Contributing

This is a personal project for educational purposes. If you find bugs or have suggestions, feel free to open an issue!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of the [texasholdem](https://github.com/SirRender00/texasholdem) Python library
- Inspired by DeepMind's DQN paper (Mnih et al., 2015)
- Poker rules based on World Series of Poker official rules

## 📧 Contact

**Yuhao Lu**
- GitHub: [@Sky-Lu6](https://github.com/Sky-Lu6)
- LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

⭐ **Star this repo** if you found it interesting or useful!

Built with ❤️ and lots of ☕
