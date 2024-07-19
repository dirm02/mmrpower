document.addEventListener("DOMContentLoaded", function () {
    const subPopulationsInput = document.getElementById("subPopulations");
    const errorRateInput = document.getElementById("errorRate");
    const scoreTypeSelect = document.getElementById("scoreType");
    const calculateButton = document.querySelector(".calculate-button");

    calculateButton.style.display = "none";
    subPopulationsInput.addEventListener("change", handleSubPopulationsChange);
    subPopulationsInput.addEventListener("input", handleSubPopulationsChange);
    subPopulationsInput.addEventListener("input", function () {
        if (subPopulationsInput.value < 2) {
            subPopulationsInput.value = 2;
        }
    });
    errorRateInput.addEventListener("input", handleErrorRateChange);
    scoreTypeSelect.addEventListener("change", handleScoreTypeChange);
});

function handleScoreTypeChange() {
    const scoreType = document.getElementById("scoreType").value;
    const inputValues = document.getElementById("inputValues");
    const tableTitle = document.getElementById("table-title");
    const calculateButton = document.querySelector(".calculate-button");
    const resetButton = document.querySelector(".reset-button");

    if (scoreType) {
        inputValues.style.display = "table";
        tableTitle.style.display = "block";
        calculateButton.style.display = "inline-block";
        resetButton.style.display = "inline-block";
    } else {
        inputValues.style.display = "none";
        tableTitle.style.display = "none";
        calculateButton.style.display = "none";
        resetButton.style.display = "none";
    }
}

function handleSubPopulationsChange() {
    const subPopulations = parseInt(document.getElementById("subPopulations").value, 10);
    const tbody = document.getElementById("inputValuesBody");
    const currentRows = tbody.getElementsByTagName("tr").length;

    if (subPopulations < 2 || subPopulations > 10) {
        document.getElementById("inputValues").style.display = "none";
        document.querySelector(".calculate-button").style.display = "none";
        return;
    }

    if (subPopulations !== currentRows) {
        if (subPopulations > currentRows) {
            for (let i = currentRows + 1; i <= subPopulations; i++) {
                const newRow = document.createElement("tr");
                newRow.innerHTML = `
                    <td class="non-editable">${i}</td>
                    <td class="editable" contenteditable="true" data-type="number" data-min="0" data-max="100"></td>
                    <td class="editable" contenteditable="true" data-type="number" data-min="0" data-max="1"></td>
                    <td class="editable with-decimals" contenteditable="true" data-type="decimal" data-min="0"></td>
                    <td class="editable with-decimals" contenteditable="true" data-type="decimal" data-min="0"></td>
                    <td class="editable with-decimals" contenteditable="true" data-type="number" data-min="0" data-max="1"></td>
                    <td class="editable with-decimals" contenteditable="true" data-type="number" data-min="0" data-max="1"></td>
                    <td class="editable with-decimals" contenteditable="true" data-type="number" data-min="0" data-max="1"></td>`;
                tbody.appendChild(newRow);
            }
        } else {
            while (tbody.getElementsByTagName("tr").length > subPopulations) {
                tbody.removeChild(tbody.lastChild);
            }
        }
    }
}

function handleErrorRateChange() {
    const errorRateInput = document.getElementById("errorRate");
    if (errorRateInput.value < 0 || errorRateInput.value > 1) {
        errorRateInput.setCustomValidity("Value must be between 0 and 1.");
    } else {
        errorRateInput.setCustomValidity("");
    }
}

function resetForm() {
    document.getElementById("powerAnalysisForm").reset();
    document.getElementById("inputValuesBody").innerHTML = `
        <tr>
            <td class="non-editable">1</td>
            <td class="editable" contenteditable="true" data-type="number" data-min="0" data-max="100"></td>
            <td class="editable" contenteditable="true" data-type="number" data-min="0" data-max="1"></td>
            <td class="editable with-decimals" contenteditable="true" data-type="decimal" data-min="0"></td>
            <td class="editable with-decimals" contenteditable="true" data-type="decimal" data-min="0"></td>
            <td class="editable with-decimals" contenteditable="true" data-type="number" data-min="0" data-max="1"></td>
            <td class="editable with-decimals" contenteditable="true" data-type="number" data-min="0" data-max="1"></td>
            <td class="editable with-decimals" contenteditable="true" data-type="number" data-min="0" data-max="1"></td>
        </tr>
        <tr>
            <td class="non-editable">2</td>
            <td class="editable" contenteditable="true" data-type="number" data-min="0" data-max="100"></td>
            <td class="editable" contenteditable="true" data-type="number" data-min="0" data-max="1"></td>
            <td class="editable with-decimals" contenteditable="true" data-type="decimal" data-min="0"></td>
            <td class="editable with-decimals" contenteditable="true" data-type="decimal" data-min="0"></td>
            <td class="editable with-decimals" contenteditable="true" data-type="number" data-min="0" data-max="1"></td>
            <td class="editable with-decimals" contenteditable="true" data-type="number" data-min="0" data-max="1"></td>
            <td class="editable with-decimals" contenteditable="true" data-type="number" data-min="0" data-max="1"></td>
        </tr>`;
    document.getElementById("inputValues").style.display = "none";
    document.getElementById("table-title").style.display = "none";
    document.querySelector(".calculate-button").style.display = "none";
    document.querySelector(".reset-button").style.display = "none";
    document.getElementById("result").style.display = "none";
    document.getElementById("result-title").style.display = "none";
    document.getElementById('subPopulations').disabled = false;
    document.getElementById('scoreType').disabled = false;
    document.getElementById('errorRate').disabled = false;
}

function calculatePower() {
    const errorRate = parseFloat(document.getElementById('errorRate').value);
    const tableBody = document.getElementById('inputValuesBody');
    const rows = tableBody.getElementsByTagName('tr');
    const scoreType = document.getElementById('scoreType').value;

    let allCellsFilled = true;

    for (let i = 0; i < rows.length; i++) {
        const cells = rows[i].getElementsByTagName('td');

        for (let j = 1; j < cells.length; j++) {
            const cellValue = cells[j].innerText.trim();

            if (cellValue === "") {
                cells[j].style.backgroundColor = "red";
                allCellsFilled = false;
            } else {
                cells[j].style.backgroundColor = "";
            }
        }
    }

    if (!allCellsFilled) {
        alert("Please fill all cells before calculating power.");
        return;
    }

    let sampleSizes = [];
    let xyCorrelations = [];
    let sdX = [];
    let sdY = [];
    let reliabilityX = [];
    let reliabilityY = [];
    let truncationProportions = [];

    for (let i = 0; i < rows.length; i++) {
        const cells = rows[i].getElementsByTagName('td');
        sampleSizes.push(parseFloat(cells[1].innerText));
        xyCorrelations.push(parseFloat(cells[2].innerText));
        sdX.push(parseFloat(cells[3].innerText));
        sdY.push(parseFloat(cells[4].innerText));
        reliabilityX.push(parseFloat(cells[5].innerText));
        reliabilityY.push(parseFloat(cells[6].innerText));
        truncationProportions.push(parseFloat(cells[7].innerText));
    }

    let sigxobs, sigyobs, rhoobs;

    if (scoreType === 'observable') {
        sigxobs = sdX;
        sigyobs = sdY;
        rhoobs = xyCorrelations;
    } else if (scoreType === 'true') {
        sigxobs = sdX.map((v, i) => v / Math.sqrt(reliabilityX[i]));
        sigyobs = sdY.map((v, i) => v / Math.sqrt(reliabilityY[i]));
        rhoobs = xyCorrelations.map((v, i) => v * Math.sqrt(reliabilityX[i] * reliabilityY[i]));
    }

    const alpha = errorRate;
    const k = sampleSizes.length;
    const n = sampleSizes;
    const sigx_obs = sigxobs;
    const sigy_obs = sigyobs;
    const alphax = reliabilityX;
    const alphay = reliabilityY;
    const rho_obs = rhoobs;
    const TruncProp = truncationProportions;

    const data = {
        alpha,
        k,
        n,
        sigx_obs,
        sigy_obs,
        alphax,
        alphay,
        rho_obs,
        TruncProp
    };

    fetch('http://127.0.0.1:5000/calculate_power', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
        .then(response => {
            if (!response.ok) {
                return response.json().then(errorData => {
                    throw new Error(errorData.error);
                });
            }
            return response.json();
        })
        .then(result => {
            if (result.power === null || isNaN(result.power)) {
                // displayResults(data, NaN) // In MATLAB, it gives 1 for the case "Inf or NaN value encountered. " in the internal calculation process.
                throw new Error('Calculation resulted in NaN');
            }
            power_obs = result.power.toPrecision(2)
            displayResults(data, power_obs);
        })
        .catch(error => {
            console.error('Error:', error);
            displayResults(data, NaN);
            //alert(`Error: ${error.message}`);
        });

    document.getElementById('subPopulations').disabled = true;
    document.getElementById('scoreType').disabled = true;
    document.getElementById('errorRate').disabled = true;
}

function displayResults(params_obs, pow_obs) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = '';

    const table = document.createElement('table');
    table.classList.add('result-table');
    const thead = document.createElement('thead');
    const tbody = document.createElement('tbody');

    const headerRow1 = document.createElement('tr');

    const subgroupHeader = document.createElement('th');
    subgroupHeader.innerText = 'Subgroup';
    subgroupHeader.rowSpan = 2;
    headerRow1.appendChild(subgroupHeader);

    const sampleSizesHeader = document.createElement('th');
    sampleSizesHeader.innerText = 'Sample Sizes';
    sampleSizesHeader.rowSpan = 2;
    headerRow1.appendChild(sampleSizesHeader);

    const xyCorrelationHeader = document.createElement('th');
    xyCorrelationHeader.innerText = 'XY Correlation';
    xyCorrelationHeader.rowSpan = 2;
    headerRow1.appendChild(xyCorrelationHeader);

    const sdHeader = document.createElement('th');
    sdHeader.innerText = 'Standard Deviation';
    sdHeader.colSpan = 2;
    headerRow1.appendChild(sdHeader);

    const reliabilityHeader = document.createElement('th');
    reliabilityHeader.innerText = 'Reliability';
    reliabilityHeader.colSpan = 2;
    headerRow1.appendChild(reliabilityHeader);

    const truncationProportionHeader = document.createElement('th');
    truncationProportionHeader.innerText = 'Truncation Proportion';
    truncationProportionHeader.rowSpan = 2;
    headerRow1.appendChild(truncationProportionHeader);

    thead.appendChild(headerRow1);

    const headerRow2 = document.createElement('tr');

    const sdXHeader = document.createElement('th');
    sdXHeader.innerText = 'X';
    headerRow2.appendChild(sdXHeader);

    const sdYHeader = document.createElement('th');
    sdYHeader.innerText = 'Y';
    headerRow2.appendChild(sdYHeader);

    const reliabilityXHeader = document.createElement('th');
    reliabilityXHeader.innerText = 'X';
    headerRow2.appendChild(reliabilityXHeader);

    const reliabilityYHeader = document.createElement('th');
    reliabilityYHeader.innerText = 'Y';
    headerRow2.appendChild(reliabilityYHeader);

    thead.appendChild(headerRow2);

    table.appendChild(thead);

    for (let i = 0; i < params_obs.k; i++) {
        const row = document.createElement('tr');

        const subgroupCell = document.createElement('td');
        subgroupCell.innerText = i + 1;
        row.appendChild(subgroupCell);

        const sampleSizeCell = document.createElement('td');
        sampleSizeCell.innerText = params_obs.n[i];
        row.appendChild(sampleSizeCell);

        const xyCorrelationCell = document.createElement('td');
        xyCorrelationCell.innerText = params_obs.rho_obs[i].toFixed(3);
        row.appendChild(xyCorrelationCell);

        const sdXCell = document.createElement('td');
        sdXCell.innerText = params_obs.sigx_obs[i].toFixed(3);
        sdXCell.classList.add('with-decimals');
        row.appendChild(sdXCell);

        const sdYCell = document.createElement('td');
        sdYCell.innerText = params_obs.sigy_obs[i].toFixed(3);
        sdYCell.classList.add('with-decimals');
        row.appendChild(sdYCell);

        const reliabilityXCell = document.createElement('td');
        reliabilityXCell.innerText = params_obs.alphax[i].toFixed(3);
        reliabilityXCell.classList.add('with-decimals');
        row.appendChild(reliabilityXCell);

        const reliabilityYCell = document.createElement('td');
        reliabilityYCell.innerText = params_obs.alphay[i].toFixed(3);
        reliabilityYCell.classList.add('with-decimals');
        row.appendChild(reliabilityYCell);

        const truncationProportionCell = document.createElement('td');
        truncationProportionCell.innerText = params_obs.TruncProp[i].toFixed(3);
        row.appendChild(truncationProportionCell);

        tbody.appendChild(row);
    }

    table.appendChild(tbody);

    resultDiv.appendChild(table);

    const powerText = document.createElement('p');
    powerText.innerText = 'The power to detect the moderating effect is ' + pow_obs + ".";
    resultDiv.appendChild(powerText);
    document.getElementById("result-title").style.display = "block";
    resultDiv.style.display = 'block';
}

function updateMinMaxTooltips() {
    const editableCells = document.querySelectorAll('.editable');
    const headers = document.querySelectorAll('thead th');

    editableCells.forEach(cell => {
        const dataType = cell.getAttribute('data-type');
        const minVal = parseFloat(cell.getAttribute('data-min'));
        const maxVal = parseFloat(cell.getAttribute('data-max'));

        let tooltipText = `Min: ${minVal}`;

        if (dataType === 'decimal') {
            tooltipText += ' (decimal)';
        }

        if (!isNaN(maxVal)) {
            tooltipText += `, Max: ${maxVal}`;
        } else {
            tooltipText += ', Max: No limit';
        }

        const columnIndex = cell.cellIndex;
        const headerText = headers[columnIndex].innerText.trim();

        if (headerText === 'Standard Deviation X' || headerText === 'Standard Deviation Y') {
            cell.title = `Min: ${minVal}`;
        } else {
            cell.title = tooltipText;
        }
    });
}

document.addEventListener('mouseover', function (event) {
    if (event.target.classList.contains('editable')) {
        updateMinMaxTooltips();
    }
});

document.addEventListener('input', function (event) {
    if (event.target.classList.contains('editable')) {
        validateCell(event.target);
    }
});

function validateCell(cell) {
    const value = cell.innerText.trim();
    const dataType = cell.getAttribute('data-type');
    const minValue = parseFloat(cell.getAttribute('data-min'));
    const maxValue = parseFloat(cell.getAttribute('data-max'));

    let isValid = true;

    if (dataType === 'number' || dataType === 'decimal') {
        const numericValue = parseFloat(value);

        if (isNaN(numericValue) || numericValue < minValue || (!isNaN(maxValue) && numericValue > maxValue)) {
            isValid = false;
        }

        if (dataType === 'decimal' && !/^\d*\.?\d+$/.test(value)) {
            isValid = false;
        }
    }

    cell.style.backgroundColor = isValid ? '' : 'red';
}

function validateTable() {
    const editableCells = document.querySelectorAll('.editable');
    let allValid = true;

    editableCells.forEach(cell => {
        const isValid = validateCell(cell);
        if (!isValid) {
            allValid = false;
        }
    });

    return allValid;
}
