// autoplay carousel
(function(){
  const slides = document.querySelectorAll('.slide');
  const next = document.getElementById('next');
  const prev = document.getElementById('prev');
  let idx = 0;
  function show(i){
    slides.forEach(s => s.classList.remove('active'));
    slides[i].classList.add('active');
  }
  function nextSlide(){
    idx = (idx + 1) % slides.length; show(idx);
  }
  function prevSlide(){
    idx = (idx - 1 + slides.length) % slides.length; show(idx);
  }

  let timer = setInterval(nextSlide, 4500);
  next && next.addEventListener('click', ()=>{ nextSlide(); resetTimer();});
  prev && prev.addEventListener('click', ()=>{ prevSlide(); resetTimer();});
  function resetTimer(){ clearInterval(timer); timer = setInterval(nextSlide,4500); }

  // safe check if slides not loaded yet
  if(!slides.length){
    // fallback: remove controls
    const n = document.getElementById('next'), p = document.getElementById('prev');
    n && n.remove(); p && p.remove();
  }

})();
