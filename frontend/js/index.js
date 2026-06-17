// الحصول على العناصر
const modal = document.getElementById("myModal");
const btn = document.querySelector(".animated-button");
const closeBtn = document.querySelector(".close-btn");

// فتح المربع عند الضغط على الزرار
btn.onclick = function() {
  modal.style.display = "block";
}

// إغلاق المربع عند الضغط على (X)
closeBtn.onclick = function() {
  modal.style.display = "none";
}

// إغلاق المربع لو المستخدم ضغط في أي مكان بره المربع
window.onclick = function(event) {
  if (event.target == modal) {
    modal.style.display = "none";
  }
}

