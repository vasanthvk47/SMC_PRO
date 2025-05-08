document.addEventListener("DOMContentLoaded", function () {
    const container = document.querySelector('.container');
    const loginBtn = document.querySelector('.login-btn');
    const registerBtn = document.querySelector('.register-btn');

    loginBtn.addEventListener('click', () => {
        container.classList.remove('active');
    });

    registerBtn.addEventListener('click', () => {
        container.classList.add('active');
    });
});
