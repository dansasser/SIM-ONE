# SIM-ONE Astro Chat Interface

A modern, production-ready ChatGPT-style chat interface built with Astro 5.0+ that showcases the SIM-ONE Framework's five-agent cognitive governance pipeline.

## ✨ Features

- 🎯 **Governed Cognition**: Visual representation of SIM-ONE's five-agent system
- 🎨 **Modern UI**: ChatGPT-inspired design with dark/light themes  
- 📱 **Responsive**: Works beautifully on desktop and mobile
- ⚡ **Fast**: Astro's islands architecture for optimal performance
- 🔐 **Secure**: OAuth authentication with guest mode
- 🌐 **Real-time**: WebSocket integration for live agent pipeline updates
- ♿ **Accessible**: Full keyboard navigation and screen reader support

## 🏗️ Architecture

Built with:
- **Astro 5.0+** - Islands architecture with SSR
- **React 18** - Interactive islands for dynamic components  
- **TypeScript** - Full type safety
- **Tailwind CSS** - Utility-first styling with custom design system
- **WebSockets** - Real-time communication with SIM-ONE backend

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your SIM-ONE API settings
   ```

3. **Start Development Server**
   ```bash
   npm run dev
   ```

4. **Open in Browser**
   ```
   http://localhost:3000
   ```

## 🤖 SIM-ONE Integration

The interface connects to your SIM-ONE mCP server and displays:

- **Real-time Agent Pipeline**: Watch Ideator → Drafter → Reviser → Critic → Summarizer
- **Processing Styles**: Universal Chat, Analytical, Creative, Academic, Business, Technical
- **Priority Levels**: Fast (2-5s), Balanced (3-10s), Quality (5-20s)
- **Quality Metrics**: Coherence, creativity, accuracy scores in real-time

## 🎨 Design System

The interface uses a comprehensive design system based on the specification:

- **Colors**: Dark/light themes with SIM-ONE brand colors
- **Typography**: Inter font family with responsive scales
- **Components**: Reusable Astro components with TypeScript props
- **Animations**: Smooth transitions with reduced motion support
- **Responsive**: Mobile-first design with breakpoint system

## 📁 Project Structure

```
src/
├── components/        # Astro components
│   ├── layout/       # Header, Sidebar, Footer
│   ├── chat/         # Chat interface components
│   ├── ui/           # Reusable UI components  
│   └── icons/        # SVG icon components
├── islands/          # React interactive components
│   ├── chat/         # ChatInput, AgentPipeline
│   ├── auth/         # Authentication forms
│   └── settings/     # Settings modals
├── layouts/          # Page layouts
├── pages/            # Astro pages and API routes
├── lib/              # Utilities and types
├── styles/           # CSS and design system
└── assets/           # Images, fonts, icons
```

## 🔧 Development

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run check` - Type check TypeScript
- `npm test` - Run test suite

## 🚀 Deployment

Built for modern deployment platforms:

- **Vercel/Netlify**: Static + SSR support
- **Node.js**: Standalone server mode
- **Docker**: Container-ready
- **CloudFlare Pages**: Edge deployment

## 📝 License

Part of the SIM-ONE Framework - See main project license.