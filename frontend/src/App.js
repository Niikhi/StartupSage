import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';
import '@fontsource/poppins/300.css';
import '@fontsource/poppins/400.css';
import '@fontsource/poppins/500.css';
import '@fontsource/poppins/600.css';

function App() {
    const [question, setQuestion] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const chatEndRef = useRef(null);
    


    useEffect(() => {
        const chatHistoryDiv = document.getElementById('chat-history');
        chatHistoryDiv.scrollTop = chatHistoryDiv.scrollHeight;
    }, [chatHistory]);

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (question.trim() === '') return;

        appendUserMessage(question);
        setQuestion('');
        setIsLoading(true);

        try {
            await sendQuestion(question);
        } catch (error) {
            console.error('Error sending question:', error);
            appendBotMessage("Sorry, I encountered an error. Please try again.");
        } finally {
            setIsLoading(false);
        }
    };

    const appendUserMessage = (message) => {
        const newMessage = { user: 'user', message: message, timestamp: new Date() };
        setChatHistory(prevHistory => [...prevHistory, newMessage]);
    };

    const appendBotMessage = (message) => {
        const newMessage = { user: 'bot', message: message, timestamp: new Date() };
        setChatHistory(prevHistory => [...prevHistory, newMessage]);
    };

    const sendQuestion = async (question) => {
        try {
            const response = await axios.post('http://localhost:5001/query', { question });
            appendBotMessage(response.data.response);
        } catch (error) {
            console.error('Error sending question:', error);
            throw error;
        }
    };

    const createMessageElement = (chat) => (
        <div 
            className={`message ${chat.user}-message`}
            style={{
                opacity: 0,
                animation: 'fadeIn 0.5s forwards',
            }}
        >
            <div className="message-content">{chat.message}</div>
            <div className="message-timestamp">
                {chat.timestamp.toLocaleTimeString()}
            </div>
        </div>
    );

    return (
    <div className="container" style={{ background: 'linear-gradient(135deg, #9ab8e6 15%, #c3cfe2 50%, #9ab8e6 80%)' }}>
        <div className='title'>
        <h1 id="text" className="console-container"> Startup GPT</h1>
        </div>
        
           <div id="chat-container">
                <div id="chat-history">
                    {chatHistory.map((chat, index) => (
                        <div key={index}>
                            {createMessageElement(chat)}
                        </div>
                    ))}
                    {isLoading && <div className="typing-indicator"> wait a sec...</div>}
                    <div ref={chatEndRef} />
                </div>
                <form id="query-form" onSubmit={handleSubmit}>
                    <textarea
                        id="question"
                        value={question}
                        onChange={(e) => setQuestion(e.target.value)}
                        rows="1"
                        placeholder="Type your question here..."
                    ></textarea>
                    <button type="submit" disabled={isLoading}>Chat</button>
                </form>
            </div>
        </div>
    );
}

export default App;