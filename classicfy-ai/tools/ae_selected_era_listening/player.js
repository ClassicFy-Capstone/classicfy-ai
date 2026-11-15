/* Official YouTube IFrame API; never extract audio or mask the video surface. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.ThemePlayer = api;
})(globalThis, function () {
  "use strict";
  let apiPromise;
  function loadApi(win) {
    if (win.YT?.Player) return Promise.resolve(win.YT);
    if (apiPromise) return apiPromise;
    apiPromise = new Promise((resolve, reject) => {
      const previous = win.onYouTubeIframeAPIReady;
      const script = win.document.createElement("script");
      const timer = win.setTimeout(() => fail(), 15000);
      function fail() { win.clearTimeout(timer); script.remove(); apiPromise = null; reject(new Error("YouTube 플레이어를 불러오지 못했습니다. 원본 링크를 이용하세요.")); }
      win.onYouTubeIframeAPIReady = () => {
        win.clearTimeout(timer);
        try { if (typeof previous === "function") previous(); } catch (_) {}
        resolve(win.YT);
      };
      script.src = "https://www.youtube.com/iframe_api";
      script.onerror = fail;
      win.document.head.append(script);
    });
    return apiPromise;
  }
  function validateRange(videoId, start, end) {
    if (!/^[A-Za-z0-9_-]{11}$/.test(videoId) || !Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start) throw new Error("Invalid video or excerpt boundaries");
    return {videoId, startSeconds:start, endSeconds:end};
  }
  function errorMessage(code) {
    if (code === 101 || code === 150) return "이 영상은 YouTube가 외부 재생을 제한합니다. 원본 링크로 들어주세요.";
    if (code === 100) return "이 영상은 삭제되었거나 공개되지 않았습니다.";
    if (code === 153) return "YouTube가 재생 페이지 정보를 확인하지 못했습니다. HTTP 서버 주소로 열거나 원본 링크를 이용하세요.";
    return "YouTube 삽입 재생에 실패했습니다. 원본 링크를 이용하세요.";
  }
  function createPlayer({api, element, videoId, start, end, origin, repeat = () => false, onStatus = () => {}, onPlay = () => {}, onError = () => {}, onRepeat = () => {}, schedule = setInterval, cancel = clearInterval}) {
    let range = validateRange(videoId, start, end), ready = false, disposed = false, desired = false, started = false, failed = false, cycling = false, completed = false, lastStatus = "loading";
    const player = new api.Player(element, {
      width:"100%", height:"100%", videoId,
      playerVars:{origin, controls:0, disablekb:1, fs:0, playsinline:1, autoplay:0, rel:0},
      events:{
        onReady:() => { if (disposed) return; ready = true; desired ? load() : (player.cueVideoById({...range}), status("ready")); },
        onStateChange:event => {
          if (disposed || failed) return;
          if (event.data === 1) { cycling = false; completed = false; desired = true; started = true; onPlay(); status("playing"); }
          else if (event.data === 2 && !cycling && !completed) { desired = false; status("paused"); }
          else if (event.data === 0 && desired) finish();
        },
        onError:event => { if (disposed) return; failed = true; desired = false; status("error"); onError(event.data, errorMessage(event.data)); },
        onAutoplayBlocked:() => { if (!disposed) { desired = false; cycling = false; status("autoplay-blocked"); } }
      }
    });
    function status(value) { lastStatus = value; onStatus(value); }
    function load() { cycling = true; started = true; player.loadVideoById({...range}); }
    function finish() {
      if (disposed || failed || cycling || !desired) return;
      if (repeat()) { onRepeat(); status("repeating"); load(); }
      else { desired = false; started = false; completed = true; player.pauseVideo(); player.cueVideoById({...range}); status("ended"); }
    }
    const timer = schedule(() => {
      if (!disposed && ready && !failed && desired && !cycling && player.getPlayerState() === 1 && player.getCurrentTime() >= range.endSeconds) finish();
    }, 100);
    return {
      play() { if (disposed || failed) return; desired = true; completed = false; if (ready) started ? player.playVideo() : load(); },
      pause() { desired = false; cycling = false; if (ready && !disposed) player.pauseVideo(); },
      setRange(from, to) { range = validateRange(videoId, from, to); desired = false; started = false; cycling = false; completed = false; if (ready && !disposed) { player.pauseVideo(); player.cueVideoById({...range}); status("ready"); } },
      getStatus() { return lastStatus; },
      destroy() { disposed = true; desired = false; cancel(timer); player.destroy(); }
    };
  }
  return {loadApi, validateRange, errorMessage, createPlayer};
});
