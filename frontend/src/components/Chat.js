import React, { useState } from 'react';

const Chat = () => {
    const [question, setQuestion] = useState('');
    const [response, setResponse] = useState('');

    const handleSubmit = async (event) => {
        event.preventDefault();
        try {
            const res = await fetch('http://localhost:5000/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question })
            });
            const data = await res.json();
            setResponse(data.response);
        } catch (error) {
            console.error('Error fetching response:', error);
        }
    };

    return (
        <div className="chat-container">
            <form onSubmit={handleSubmit} className="chat-form">
                <textarea
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Type your question here..."
                    className="chat-input"
                />
                <button type="submit" className="chat-button">Send</button>
            </form>
            {response && <div className="chat-response">{response}</div>}
        </div>
    );
};

export default Chat;
