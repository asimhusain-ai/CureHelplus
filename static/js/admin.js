document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-progress]").forEach((el) => {
    const raw = parseFloat(el.dataset.progress);
    const percent = Number.isFinite(raw) ? Math.min(Math.max(raw, 0), 100) : 0;
    el.style.width = `${percent}%`;
    el.setAttribute("aria-valuenow", percent.toString());
  });

  document.querySelectorAll("[data-donut]").forEach((card) => {
    let data = [];
    try {
      data = JSON.parse(card.dataset.donut || "[]");
    } catch (error) {
      console.warn("Invalid donut dataset", error);
      data = [];
    }

    const donut = card.querySelector(".donut-visual");
    if (!donut) return;

    const trackColor = getComputedStyle(donut).getPropertyValue("--donut-track") || "rgba(255,255,255,0.08)";
    donut.style.background = `conic-gradient(${trackColor} 0deg 360deg)`;

    if (!data.length) return;

    let current = 0;
    const segments = data.map((item) => {
      const cleanedPercent = Math.min(Math.max(Number(item.percent) || 0, 0), 100);
      const sweep = (cleanedPercent / 100) * 360;
      const start = current;
      current += sweep;
      return `${item.color || "#22D3EE"} ${start}deg ${current}deg`;
    });

    requestAnimationFrame(() => {
      donut.style.background = `conic-gradient(${segments.join(",")})`;
    });
  });

  const chartTargets = document.querySelectorAll("canvas[data-chart]");
  if (chartTargets.length) {
    if (typeof Chart === "undefined") {
      console.warn("Chart.js is required for donut charts but is not loaded.");
    } else {
      chartTargets.forEach((canvas) => {
        let dataset = [];
        try {
          dataset = JSON.parse(canvas.dataset.chart || "[]");
        } catch (error) {
          console.warn("Invalid chart dataset", error);
          dataset = [];
        }

        if (!dataset.length) {
          return;
        }

        const labels = dataset.map((entry) => entry.label || "");
        const values = dataset.map((entry) => {
          const value = Number(entry.count);
          return Number.isFinite(value) && value > 0 ? value : 0;
        });
        const colors = dataset.map((entry, index) => entry.color || `hsl(${(index / dataset.length) * 360}, 70%, 55%)`);

        const ctx = canvas.getContext("2d");
        if (!ctx) {
          return;
        }

        new Chart(ctx, {
          type: "doughnut",
          data: {
            labels,
            datasets: [
              {
                data: values,
                backgroundColor: colors,
                borderColor: colors,
                borderWidth: 22,
                hoverBorderWidth: 22,
                borderRadius: 24,
                spacing: 8,
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: "70%",
            plugins: {
              legend: { display: false },
              tooltip: { enabled: false },
            },
            animation: {
              animateRotate: true,
              duration: 900,
            },
          },
        });
      });
    }
  }
});
