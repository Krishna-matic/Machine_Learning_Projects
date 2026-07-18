/**
 * script.js
 * ---------
 * Pure front-end interaction layer for Rainscope. This file never talks
 * to the server directly and never changes what gets submitted to
 * Flask's `/predict` route -- it only improves the experience of filling
 * in the existing form fields and displaying the existing result data.
 *
 * Responsibilities:
 *   1. Smoothly expand/collapse the "Advanced inputs" panel.
 *   2. Drive the Yes/No segmented toggle for "Did it rain today?" and
 *      keep its hidden <input name="rain_today"> in sync.
 *   3. Lightweight client-side validation with a friendly shake + focus
 *      instead of relying solely on the browser's default tooltips.
 *   4. Default the date field to today, and animate the barometer gauge
 *      on the result page.
 */

(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", () => {
    initAdvancedToggle();
    initRainTodayToggle();
    initDateDefault();
    initFormValidation();
    initGauge();
  });

  /* ------------------------------------------------------------------
   * 1. Collapsible "Advanced inputs" panel
   * ------------------------------------------------------------------ */
  function initAdvancedToggle() {
    const toggle = document.getElementById("advanced-toggle");
    const panel = document.getElementById("advanced-panel");
    if (!toggle || !panel) return;

    const label = toggle.querySelector("span");

    toggle.addEventListener("click", () => {
      const isOpen = toggle.getAttribute("aria-expanded") === "true";
      const nextOpen = !isOpen;

      toggle.setAttribute("aria-expanded", String(nextOpen));
      label.textContent = nextOpen ? "Hide advanced inputs" : "Show advanced inputs";

      if (nextOpen) {
        // Animate to the panel's natural height, then release the fixed
        // height afterwards so it still reflows correctly on resize.
        panel.style.maxHeight = panel.scrollHeight + "px";
        panel.addEventListener(
          "transitionend",
          function onOpen() {
            panel.style.maxHeight = "none";
            panel.removeEventListener("transitionend", onOpen);
          },
          { once: true }
        );
      } else {
        // Lock in the current pixel height first so the transition has
        // something concrete to animate down from.
        panel.style.maxHeight = panel.scrollHeight + "px";
        requestAnimationFrame(() => {
          panel.style.maxHeight = "0";
        });
      }
    });
  }

  /* ------------------------------------------------------------------
   * 2. Yes/No segmented toggle -> hidden `rain_today` input
   * ------------------------------------------------------------------ */
  function initRainTodayToggle() {
    const toggle = document.querySelector(".toggle");
    if (!toggle) return;

    const hiddenInput = toggle.parentElement.querySelector('input[name="rain_today"]');
    const buttons = toggle.querySelectorAll(".toggle__btn");

    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        buttons.forEach((b) => b.classList.remove("is-active"));
        btn.classList.add("is-active");
        hiddenInput.value = btn.dataset.value;
        toggle.classList.remove("has-error");
      });
    });
  }

  /* ------------------------------------------------------------------
   * 3. Default the date field to today (purely a convenience default;
   *    the user can still change it before submitting).
   * ------------------------------------------------------------------ */
  function initDateDefault() {
    const dateInput = document.querySelector('input[name="date"]');
    if (!dateInput || dateInput.value) return;

    const today = new Date();
    const iso = today.toISOString().slice(0, 10);
    dateInput.value = iso;
    dateInput.max = iso; // forecasting "tomorrow" from a future date doesn't make sense
  }

  /* ------------------------------------------------------------------
   * 4. Friendly client-side validation on submit
   * ------------------------------------------------------------------ */
  function initFormValidation() {
    const form = document.getElementById("forecast-form");
    if (!form) return;

    form.addEventListener("submit", (event) => {
      const invalidFields = Array.from(form.querySelectorAll("[required]")).filter(
        (el) => !el.value
      );

      const rainTodayInput = form.querySelector('input[name="rain_today"]');
      const rainToggle = form.querySelector(".toggle");
      const rainTodayMissing = rainToggle && !rainTodayInput.value;

      if (invalidFields.length === 0 && !rainTodayMissing) {
        return; // let the browser submit normally to /predict
      }

      event.preventDefault();

      invalidFields.forEach((el) => flashInvalid(el.closest(".field") || el));
      if (rainTodayMissing) {
        flashInvalid(rainToggle);
        rainToggle.classList.add("has-error");
      }

      // Scroll and focus the first problem field for a fast fix.
      const firstProblem = rainTodayMissing ? rainToggle : invalidFields[0];
      if (firstProblem) {
        firstProblem.scrollIntoView({ behavior: "smooth", block: "center" });
        if (typeof firstProblem.focus === "function") {
          setTimeout(() => firstProblem.focus({ preventScroll: true }), 300);
        }
      }
    });
  }

  function flashInvalid(el) {
    if (!el) return;
    el.classList.add("is-shaking");
    el.addEventListener("animationend", () => el.classList.remove("is-shaking"), { once: true });
  }

  /* ------------------------------------------------------------------
   * 5. Animate the barometer-style confidence gauge on the result page
   * ------------------------------------------------------------------ */
  function initGauge() {
    const gauge = document.querySelector(".gauge");
    if (!gauge) return;

    const confidence = Math.max(0, Math.min(100, parseFloat(gauge.dataset.confidence) || 0));

    const fill = gauge.querySelector(".gauge__fill");
    const needleGroup = gauge.querySelector(".gauge__needle-group");
    if (!fill || !needleGroup) return;

    // The arc's full stroke-dasharray length is 270 (see style.css); an
    // offset of 0 means fully drawn, 270 means fully hidden.
    const arcLength = 270;
    const targetOffset = arcLength - (arcLength * confidence) / 100;

    // The needle sweeps a 180-degree arc: -90deg (0%) to +90deg (100%).
    const targetAngle = -90 + (confidence / 100) * 180;

    // Kick off the transition on the next frame so the CSS transition
    // (defined in style.css) actually animates from the initial state
    // rather than snapping straight to the final value.
    requestAnimationFrame(() => {
      fill.style.strokeDashoffset = String(targetOffset);
      needleGroup.style.transform = `rotate(${targetAngle}deg)`;
    });
  }
})();
