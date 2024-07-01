document.addEventListener('DOMContentLoaded', function() {
    var chatHistory = document.getElementById('chat-history');
    var queryForm = document.getElementById('query-form');
    var questionInput = document.getElementById('question');

    queryForm.addEventListener('submit', function(event) {
        event.preventDefault();
        var question = questionInput.value.trim();

        if (question === '') {
            return;
        }

        appendUserMessage(question);
        sendQuestion(question);
        questionInput.value = '';
    });

    function appendUserMessage(message) {
        var messageElement = createMessageElement(message, 'user-message');
        chatHistory.appendChild(messageElement);
        scrollToBottom();
    }

    function appendBotMessage(message) {
        var messageElement = createMessageElement(message, 'bot-message');
        chatHistory.appendChild(messageElement);
        scrollToBottom();
    }

    function createMessageElement(message, className) {
        var messageDiv = document.createElement('div');
        messageDiv.classList.add('message', className);
        messageDiv.innerHTML = message;
        return messageDiv;
    }

    function scrollToBottom() {
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function sendQuestion(question) {
        fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded'
            },
            body: 'question=' + encodeURIComponent(question)
        })
        .then(response => response.json())
        .then(data => {
            appendBotMessage(data.response);
            showBotResponse(data.response); // New function to style bot's response
        })
        .catch(error => {
            console.error('Error sending question:', error);
        });
    }

    function showBotResponse(response) {
        var responseContainer = createMessageElement(response, 'bot-message response-container');
        chatHistory.appendChild(responseContainer);
        scrollToBottom();
    }
});
