async function sendMessage() {
    let userInput = document.getElementById("user_input").value;
    if (!userInput) return;

    let chatBox = document.getElementById("chat-box");

    // Phát âm thanh gửi tin nhắn
    document.getElementById("send-sound").play();

    // Hiển thị tin nhắn người dùng
    let userMessage = document.createElement("div");
    userMessage.classList.add("message", "user-message");
    userMessage.innerHTML = `<strong>Bạn:</strong> ${userInput}`;
    chatBox.appendChild(userMessage);
    chatBox.scrollTop = chatBox.scrollHeight;

    // Xóa ô nhập
    document.getElementById("user_input").value = "";

    // Hiển thị tin nhắn chờ xử lý
    let botMessage = document.createElement("div");
    botMessage.classList.add("message", "bot-message");
    botMessage.innerHTML = `<strong>Chatbot:</strong> 🤔 Đang suy nghĩ...`;
    chatBox.appendChild(botMessage);
    chatBox.scrollTop = chatBox.scrollHeight;

    // Gửi yêu cầu đến API
    let response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_input: userInput })
    });

    let data = await response.json();
    
    // Cập nhật tin nhắn chatbot
    setTimeout(() => {
        botMessage.innerHTML = `<strong>Chatbot:</strong> ${data.response} 😊`;
        chatBox.scrollTop = chatBox.scrollHeight;

        // Phát âm thanh nhận tin nhắn
        document.getElementById("receive-sound").play();
    }, 1000);
}
