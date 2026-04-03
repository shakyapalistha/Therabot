class AIService {
    constructor(){
        this.baseUrl = 'http://127.0.0.1:5000/chat';
        /*this.baseUrl = '/chat';*/ 
    }


    async sendMessage(message){
        try{
            const response=await fetch(this.baseUrl,{
                method:'POST',
                headers:{
                    'Content-Type':'application/json',
                    /*'Authorization':`Bearer ${this.apiKey}`*/
                },

                body:JSON.stringify({
                    message:message
                  /*  model:'gpt-4o',
                    messages: [
                        {
                            role:'system',
                            content:'You are a helpful assistant.Answer as conciesly as possible'
                        },
                        {
                            role: 'user',
                            content: message
                        }
                    ],
                    max_tokens: 1000,
                    temperature:0.7, */ 
                })
            });

            if(!response.ok){
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const data=await response.json();
            return data.response;
            /*return data.choices[0].message.content;*/

        }catch(error){
            console.error('Error sending message:',error);
            throw error;
        }
    }
}

class ChatApp{
    constructor(){
        this.aiService = new AIService();
        this.chatContainer = document.getElementById('chatContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.isLoading = false;

        this.initEventListeners();
        this.adjustTextAreaHeight();
    }

    initEventListeners(){
        this.sendButton.addEventListener('click',() => this.sendMessage());
        this.messageInput.addEventListener('keydown',(e) =>{
            if(e.key === 'Enter' && !e.shiftKey){
                e.preventDefault();
                this.sendMessage();
            }
        });
        this.messageInput.addEventListener('input', () => this.adjustTextAreaHeight());
    }

    adjustTextAreaHeight(){
        const textArea=this.messageInput;
        textArea.style.height ='auto';
        textArea.style.height = Math.min(textArea.scrollHeight,120)+'px';
    }
    async sendMessage(){
        const message = this.messageInput.value.trim();
        if(!message || this.isLoading)  return;

        this.addMessage(message, true);
        this.messageInput.value = '';
        this.adjustTextAreaHeight();
        this.setLoading(true);
        this.showTypingIndicator();

        try{
            const response = await this.aiService.sendMessage(message);
            this.hideTypingIndicator();
            this.addMessage(response, false);
        }catch (error){
            this.hideTypingIndicator();
            this.addMessage("I apologize ,but I'm having trouble now please try again ",false);
        }finally{
            this.setLoading(false);
        }
    }
    addMessage(text,isUser){
        const messageDiv = document.createElement('div');
        messageDiv.className=`message ${isUser ? 'user-message' :'ai-message'}`;

        const avatar = document.createElement('div');
        avatar.className = `message-avatar ${isUser ? 'user-message-avatar' :'ai-message-avatar'}`;
        avatar.innerHTML = isUser ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>'

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent =text;

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(bubble);

        this.chatContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    showTypingIndicator(){
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing-indicator';
        typingDiv.id = 'typingIndicator';

        const avatar = document.createElement('div');
        avatar.className='message-avatar ai-message-avatar';
        avatar.innerHTML ='<i class="fa-solid fa-robot"></i>';

        const bubble = document.createElement('div');
        bubble.className = 'typing-bubble';

        for(let i=0; i< 3;i++){
            const dot =document.createElement('div');
            dot.className ='typing-dot';
            bubble.appendChild(dot);
        }

        typingDiv.appendChild(avatar);
        typingDiv.appendChild(bubble);
        this.chatContainer.appendChild(typingDiv);
        this.scrollToBottom();
    }

    hideTypingIndicator(){
        const typingIndicator = document.getElementById('typingIndicator');
        if(typingIndicator){
            typingIndicator.remove();
        }
    }

    setLoading(isLoading){
        this.isLoading =isLoading;
        this.sendButton.disabled = isLoading;
        this.messageInput.disabled = isLoading;
    }

    scrollToBottom(){
        setTimeout(() =>{
            this.chatContainer.scrollTop = this.chatContainer.scrollHeight;
        },100);
    }
}

document.addEventListener('DOMContentLoaded',() =>{
    new ChatApp();
});
