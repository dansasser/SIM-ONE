# SIM-ONE SDK Integration Guide

## 🎯 Overview

This Astro chat interface is **SDK-ready** and designed to seamlessly integrate with your SIM-ONE SDK when it's released. The current implementation uses mock data to showcase all the capabilities your SDK will provide.

## 🏗️ Architecture

### SDK Abstraction Layer

The interface is built around a clean SDK abstraction located in:

- **`src/lib/sdk/simone-sdk.ts`** - Complete SDK interface definitions and mock implementation
- **`src/lib/hooks/useSimoneSDK.ts`** - React hooks for easy SDK integration
- **`src/islands/providers/SDKProvider.tsx`** - Global state management and context

### Key Interfaces

```typescript
// Main SDK Interface
export interface ISimoneSDK {
  auth: IAuthService;           // Authentication
  chat: IChatService;           // Conversations & Messages  
  processing: IProcessingService; // Agent Pipeline
  models: IModelService;        // AI Model Management
  events: IEventService;        // Real-time Events
  users: IUserService;          // User Management
}
```

## 🔄 Integration Steps

### 1. Replace Mock Implementation

When your SDK is ready, simply replace the `MockSimoneSDK` class:

```typescript
// In src/lib/sdk/simone-sdk.ts
import { YourActualSIMONESDK } from '@simone/sdk';

// Replace this line:
export const SimoneSDK: ISimoneSDK = new MockSimoneSDK();

// With this:
export const SimoneSDK: ISimoneSDK = new YourActualSIMONESDK({
  apiKey: process.env.SIMONE_API_KEY,
  baseURL: process.env.SIMONE_API_URL
});
```

### 2. Update Environment Variables

Add your actual API configuration:

```bash
# .env
SIMONE_API_URL=https://api.simone.ai
SIMONE_API_KEY=your-actual-api-key
OAUTH_CLIENT_ID=your-oauth-client-id
OAUTH_CLIENT_SECRET=your-oauth-secret
```

### 3. SDK Methods Already Implemented

The interface expects these methods from your SDK:

#### Authentication
- `auth.login(credentials)` - User login
- `auth.register(data)` - User registration  
- `auth.logout()` - User logout
- `auth.getCurrentUser()` - Get current user
- `auth.isAuthenticated()` - Check auth status

#### Chat Management
- `chat.getConversations()` - List conversations
- `chat.createConversation(title?)` - Create new chat
- `chat.getMessages(conversationId)` - Get chat messages
- `chat.sendMessage(conversationId, content, options)` - Send message

#### Processing Pipeline  
- `processing.startProcessing(input, style, priority)` - Start SIM-ONE processing
- `processing.getJob(jobId)` - Get processing job status
- `processing.subscribeToJob(jobId, callback)` - Real-time updates

#### Models & Configuration
- `models.getAvailableModels()` - List available AI models
- `models.getCurrentModel()` - Get active model
- `models.switchModel(modelId)` - Change active model

#### Real-time Events
- `events.connect()` - Establish WebSocket connection
- `events.on(event, callback)` - Subscribe to events
- `events.emit(event, data)` - Emit events

## 📊 Data Models

### Processing Styles
The interface supports these processing styles:
- `universal_chat` - General conversation
- `analytical_article` - Technical analysis  
- `creative_writing` - Creative content
- `academic_paper` - Academic writing
- `business_report` - Business communication
- `technical_documentation` - Technical guides

### Priority Levels
- `fast` - 2-5 seconds, speed optimized
- `balanced` - 3-10 seconds, balanced quality/speed  
- `quality` - 5-20 seconds, maximum quality

### Agent Types
The five-agent pipeline:
- `ideator` - Generates concepts and ideas
- `drafter` - Creates structured drafts
- `reviser` - Refines and improves content
- `critic` - Evaluates quality and coherence
- `summarizer` - Produces final output

## 🎨 UI Components Ready for SDK

### Real-time Features
- **Agent Pipeline Visualization** - Shows processing steps in real-time
- **Processing Indicators** - Live status updates during generation
- **Quality Metrics** - Displays coherence, creativity, accuracy scores
- **Message Metadata** - Shows which agents processed each response

### Interactive Elements
- **Style Selector** - Choose processing approach
- **Priority Selector** - Balance speed vs quality
- **Model Switcher** - Change active AI model
- **Conversation Management** - Create, delete, organize chats

## 🔗 React Hooks Usage

The interface provides ready-to-use React hooks:

```typescript
// Authentication
const { user, login, logout, isAuthenticated } = useAuth();

// Conversations
const { conversations, createConversation } = useConversations();

// Messages  
const { messages, sendMessage } = useMessages(conversationId);

// Processing
const { activeJobs, startProcessing } = useProcessing();

// Models
const { models, currentModel, switchModel } = useModels();

// Connection Status
const { isConnected, status } = useSDKStatus();
```

## 🚀 Demo Features

The current mock implementation demonstrates:

1. **Complete Chat Flow** - Send message → Agent processing → Response
2. **Agent Pipeline** - Visual representation of Ideator → Drafter → Reviser → Critic → Summarizer
3. **Quality Metrics** - Coherence, creativity, accuracy scores
4. **Processing Styles** - Different modes for different content types
5. **Priority Levels** - Speed vs quality tradeoffs
6. **Real-time Updates** - Live processing status and progress
7. **Conversation Management** - Multiple chat sessions
8. **Responsive Design** - Works on all devices
9. **Theming System** - Dark/light modes with SIM-ONE branding

## 🎯 Benefits of This Architecture

### For SDK Development
- **Clear Interface Contract** - Defines exactly what the SDK needs to implement
- **Mock Testing** - Full functionality testing before SDK completion
- **Type Safety** - TypeScript ensures correct implementation
- **Real-world Usage** - Interface patterns match actual use cases

### For Integration  
- **Drop-in Replacement** - Just swap mock for real SDK
- **No UI Changes** - All components work with either implementation
- **Incremental Migration** - Can replace services one by one
- **Error Handling** - Robust error boundaries and fallbacks

### For Demonstration
- **Full Showcase** - Demonstrates every SDK capability
- **Interactive Demo** - Users can experience SIM-ONE features
- **Performance Preview** - Shows real-world usage patterns
- **Marketing Ready** - Professional, polished interface

## 📝 Next Steps

1. **Complete Your SDK** - Implement the interfaces defined here
2. **Test Integration** - Replace mock with SDK in development  
3. **Add Authentication** - Implement OAuth flows
4. **Real-time Events** - Add WebSocket support for live updates
5. **Performance Optimization** - Add caching and optimization
6. **Production Deploy** - Launch with full SDK integration

## 🤝 Perfect Partnership

This interface serves as both:
- **SDK Development Guide** - Shows exactly what to build
- **Ready Integration** - Drop-in compatibility when SDK is ready  
- **Compelling Demo** - Showcases SIM-ONE's full potential
- **User Experience** - Professional, intuitive chat interface

The mock implementation ensures the UI is battle-tested and ready for your SDK from day one!