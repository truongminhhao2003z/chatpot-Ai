// Bắt sự kiện khi nhấn Enter
document.getElementById("user_input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendMessage();
    }
});

// Hàm gửi tin nhắn
async function sendMessage() {
    let userInput = document.getElementById("user_input").value.trim();
    if (!userInput) return;

    // Phát âm thanh gửi tin nhắn
    document.getElementById("send-sound").play();

    // Hiển thị tin nhắn người dùng
    addMessage("Bạn", userInput, "user-message");

    // Xóa ô nhập và vô hiệu hóa tạm thời
    document.getElementById("user_input").value = "";
    document.getElementById("user_input").disabled = true;

    // Hiển thị tin nhắn chờ xử lý
    let botMessageElement = addMessage("Chatbot", "🤔 Đang suy nghĩ...", "bot-message"); // Đổi tên biến để rõ ràng hơn

    try {
        let response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            // Truyền conversationHistory
            body: JSON.stringify({ user_input: userInput, conversation_history: conversationHistory })
        });

        let data = await response.json();

        // Cập nhật nội dung tin nhắn bot
        // Sử dụng setTimeout để giả lập độ trễ xử lý của bot (nếu bạn muốn giữ hiệu ứng này)
        setTimeout(() => {
            botMessageElement.innerHTML = `<strong>Chatbot:</strong> ${data.response} 😊`; // Sử dụng botMessageElement
            document.getElementById("receive-sound").play();
            chatBox.scrollTop = chatBox.scrollHeight;
            triggerBubbles();
            nobitaWave();

            // *** THÊM PHẦN NÀY ĐỂ HIỂN THỊ NÚT FEEDBACK ***
            if (data.conversation_id) {
                appendFeedbackControls(botMessageElement, data.conversation_id);
            }
            // *** KẾT THÚC PHẦN THÊM ***

        }, 1000);

        // Cập nhật lịch sử hội thoại (nếu API của bạn trả về)
        conversationHistory = data.conversation_history || [];

    } catch (error) {
        console.error('Error in sendMessage:', error); // Log lỗi để dễ debug
        botMessageElement.innerHTML = `<strong>Chatbot:</strong> ❌ Lỗi kết nối, vui lòng thử lại!`;
    } finally {
        document.getElementById("user_input").disabled = false;
    }
}

// Hàm thêm tin nhắn vào khung chat (cần sửa lại hàm này)
function addMessage(sender, text, className) {
    let chatBox = document.getElementById("chat-box");
    let message = document.createElement("div");
    message.classList.add("message", className);
    message.innerHTML = `<strong>${sender}:</strong> ${text}`;
    chatBox.appendChild(message);
    chatBox.scrollTop = chatBox.scrollHeight;
    return message;
}

// *** THÊM HÀM MỚI ĐỂ HIỂN THỊ VÀ GỬI FEEDBACK ***
function appendFeedbackControls(messageElement, conversationId) {
    const feedbackControls = document.createElement('div');
    feedbackControls.classList.add('feedback-controls'); // Thêm class cho CSS

    const likeBtn = document.createElement('button');
    likeBtn.classList.add('feedback-btn', 'like-btn');
    likeBtn.innerHTML = '👍'; // Icon Like
    likeBtn.title = 'Phản hồi tốt';
    likeBtn.onclick = () => sendFeedback(conversationId, 1, messageElement);

    const dislikeBtn = document.createElement('button');
    dislikeBtn.classList.add('feedback-btn', 'dislike-btn');
    dislikeBtn.innerHTML = '👎'; // Icon Dislike
    dislikeBtn.title = 'Phản hồi kém';
    dislikeBtn.onclick = () => sendFeedback(conversationId, -1, messageElement);

    feedbackControls.appendChild(likeBtn);
    feedbackControls.appendChild(dislikeBtn);
    messageElement.appendChild(feedbackControls);
}

async function sendFeedback(conversationId, score, messageElement) {
    try {
        const response = await fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conversation_id: conversationId,
                score: score
                // comment: (nếu bạn muốn cho phép người dùng nhập bình luận)
            })
        });

        if (response.ok) {
            const data = await response.json();
            console.log('Feedback sent:', data.message);
            // Vô hiệu hóa nút và hiển thị thông báo sau khi gửi
            const feedbackControls = messageElement.querySelector('.feedback-controls');
            if (feedbackControls) {
                feedbackControls.innerHTML = score === 1 ? 'Cảm ơn phản hồi tích cực của bạn!' : 'Cảm ơn phản hồi của bạn!';
                feedbackControls.style.fontSize = '0.8em';
                feedbackControls.style.marginTop = '5px';
            }
        } else {
            const errorData = await response.json();
            console.error('Error sending feedback:', errorData);
            alert(`Lỗi gửi phản hồi: ${errorData.detail || 'Không xác định.'}`);
        }
    } catch (error) {
        console.error('Network error sending feedback:', error);
        alert('Lỗi kết nối khi gửi phản hồi.');
    }
}

// Hiệu ứng Nobita vẫy tay
function nobitaWave() {
    let nobita = document.querySelector(".nobita");
    nobita.classList.add("wave");
    setTimeout(() => nobita.classList.remove("wave"), 2000);
}

// Hiệu ứng bong bóng bay lên khi chatbot trả lời
function triggerBubbles() {
    let container = document.getElementById("bubbles-container");
    for (let i = 0; i < 5; i++) {
        setTimeout(() => {
            let bubble = document.createElement("div");
            bubble.classList.add("bubble");
            bubble.style.left = Math.random() * 100 + "%";
            bubble.style.animationDuration = (Math.random() * 2 + 2) + "s";
            container.appendChild(bubble);
            setTimeout(() => bubble.remove(), 3000);
        }, i * 500);
    }
}

// Chuyển đổi chế độ sáng/tối
document.getElementById("theme-toggle").addEventListener("click", function() {
    let body = document.body;
    let chatContainer = document.querySelector(".chat-container");
    let input = document.querySelector("input");
    let button = document.querySelector("button");
    
    body.classList.toggle("dark-mode");
    body.classList.toggle("light-mode");
    chatContainer.classList.toggle("dark-mode");
    chatContainer.classList.toggle("light-mode");
    input.classList.toggle("dark-mode");
    input.classList.toggle("light-mode");
    button.classList.toggle("dark-mode");
    button.classList.toggle("light-mode");

    if (body.classList.contains("dark-mode")) {
        document.getElementById("theme-toggle").textContent = "☀️";
    } else {
        document.getElementById("theme-toggle").textContent = "🌙";
    }
});
