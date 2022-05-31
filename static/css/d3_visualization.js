const {
    csv,
    select,
    scaleLinear,
    extent,
    axisLeft,
    axisBottom,
} = d3;

// set dimensions/ margins of the graph
var margin = {top: 20, right: 20, bottom: 40, left: 50}
const width = 700;
const height = 500;  // 500 for monitor, 325 for laptop

var click_count = 0

var clicks_highlight = 0
var click1
var click2

// placement of chart
var SVG = d3.select("#dataviz_axisZoom")
    .append("svg")
        .attr("width", width + margin.left + margin.right )
        .attr("height", height + margin.top + margin.bottom)
        // .attr("preserveAspectRatio", "xMinYMin meet")
        // .attr("viewBox", "0 0 900 700")
        // .classed("svg-content", true)
    .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");    


//Read the data need to run local server: 'python -m http.server 8888' in console then open
d3.json("http://127.0.0.1:8080/data", function(data) {
    const xValue = (data) => data.tsne1;
    const yValue = (data) => data.tsne2;
    
    // x-axis
    var x = d3.scaleLinear()
        .domain(extent(data, xValue))
        .range([0, width - margin.right]);
    var xAxis = SVG.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x));

    // y-axis
    var y = d3.scaleLinear()
        .domain(extent(data, yValue))
        .range([height, margin.top-20]);
    var yAxis = SVG.append("g")
        .call(d3.axisLeft(y));
    
    // keep points in chart
    var clip = SVG.append("defs").append("SVG:clipPath")
        .attr("id", "clip")
        .append("SVG:rect")
        .attr("width", width - 20)
        .attr("height", height )
        .attr("x", 0)
        .attr("y", 0);

    var color = d3.scaleOrdinal()  // color for AutoOD pred: orange for out blue for in
        .domain([1, 0])
        .range(["#FFA500", "SteelBlue"])

    var stroke = d3.scaleOrdinal()
        .domain(["yes", "no"])
        .range(["none", "black"])

    var zoom = d3.zoom()  // zoom function: from https://www.d3-graph-gallery.com/graph/interactivity_zoom.html
        .on("zoom", updateChart);
        
    var transform = d3.zoomIdentity.translate(75, 75).scale(.8);  // starting zoom
    
    // Legend
    //SVG.append("circle").style("background-color", "white").attr("cx",710).attr("cy",15).attr("r", 6).style("fill", "SteelBlue")
    //SVG.append("circle").style("background-color", "white").attr("cx",710).attr("cy",30).attr("r", 6).style("fill", "#FFA500")
    //SVG.append("text").style("background-color", "white").attr("x", 700).attr("cy",15).text("AutoOD Prediction").style("font-size", "20px").attr("alignment-baseline","middle")
    //SVG.append("text").style("background-color", "white").attr("x", 730).attr("y", 15).text("Inlier").style("font-size", "15px").attr("alignment-baseline","middle")
    //SVG.append("text").style("background-color", "white").attr("x", 730).attr("y", 30).text("Outlier").style("font-size", "15px").attr("alignment-baseline","middle")

    // tooltip
    var tooltip = d3.select("#dataviz_axisZoom")
        .append("div")
            .attr("id", "tooltip")
            .style("position", "absolute")
            .style("opacity", 0)
            .style("background-color", "white")
            .style("border", "solid")
            .style("border-width", "1px")
            .style("border-radius", "5px")
            .style("padding", "10px")
            .style("z-index", "50")
            .style("z-index", "2")
            .style("width", "200px")
    
    var mouseover = function(d) {
        tooltip
        .style("opacity", 1)
    }
    
    function tooltip_int2str(col){
        if(col == 1){
            return "Outlier"
        }
        else if (col == 0){
            return "Inlier"
        }
        else{
            return ""
        }
    }

    function y_n_int2str(col){
        if(col == 1){
            return "Yes"
        }
        else{
            return "No"
        }
    }

    var mousemove = function(d) {
        tooltip
            .html(
                "Correct Prediction?:" + " " + y_n_int2str(d.correct) + "<br>" + 
                "Ground Truth Label:" + " " + tooltip_int2str(d.label) + "<br>" + 
                "AutoOD Prediction:" + " " + tooltip_int2str(d.prediction) + "<br>" +
                "LOF:" + " " + tooltip_int2str(d.prediction_lof) + "<br>" +
                "KNN:" + " " + tooltip_int2str(d.prediction_knn) + "<br>" +
                "IF:" + " " + tooltip_int2str(d.prediction_if) + "<br>" +
                "Mahalanobis:" + " " + tooltip_int2str(d.prediction_mahalanobis) + "<br>" +
                "Reliable Label?:" + " " + y_n_int2str(d.reliable_31) + "<br>"
            
            ) 
            // location of the tooltip
            .style("left", (d3.mouse(this)[0]+70) + "px")
            .style("top", (d3.mouse(this)[1]+30) + "px")
    }

      // A function that change this tooltip when the mouse leaves a point
    var mouseleave = function(d) {
        tooltip
            .transition()
            .duration(1)
            .style("opacity", 0)
            .style("left", (d3.mouse(this)[0]-900) + "px")  // move hidden tooltip out of chart so user can interact with points covered by hidden tooltip
            .style("top", (d3.mouse(this)[1]-900) + "px")
    }
        
      // plot
    var scatter = SVG.append('g')
        .attr("clip-path", "url(#clip)")
    // marks
    scatter
        .selectAll("circle")
        .data(data)
        .enter()
        .append("circle")
            .attr("cx", function (d) {return x(d.tsne1);})
            .attr("cy", function (d) {return y(d.tsne2);})
            .attr("r", 3.5)
            .style("stroke", "white") 
            .style("fill", function (d){return color(d.prediction)})  // add color
            .style("opacity", 0.5)
            //.style("stroke", function (d){return stroke(d.correct)})
        // for tooltip
        .on("mouseover", mouseover)
        .on("mousemove", mousemove)
        .on("mouseleave", mouseleave)
        .on("click", function(d, i) {  // click on data point will highlight
            clicks_highlight = 1 + clicks_highlight
            if(clicks_highlight <= 2){
                if(click2){  // only if there has been a second click
                    scatter  // clear selection
                        .selectAll("circle")
                        .attr("r", 3.5)
                        .style("stroke", "white") 
                    click2.transition()
                        .duration('100')
                        .attr("r", 7)
                        .style("stroke", "black") 
                }
                click1 = d3.select(this)  // keep state of past selection
                click1.transition()
                    .duration('100')
                    .attr("r", 7)
                    .style("stroke", "black")
            }
            else{
                clicks_highlight = 0
                clicks_highlight = 1 + clicks_highlight
                scatter
                    .selectAll("circle")  // clear selection
                    .attr("r", 3.5)
                    .style("stroke", "white") 
                click1.transition()
                    .duration('100')
                    .attr("r", 7)
                    .style("stroke", "black") 
                click2 = d3.select(this)  // keep state of past selection
                click2.transition()
                    .duration('100')
                    .attr("r", 7)
                    .style("stroke", "black") 
            }

            click_count = 1 + click_count  // keep track of which object to update
            if(click_count% 2 == 0){
                //data_tabel1(d.id, d.att1, d.att2, d.att3, d.att4, d.att5, d.att6, d.att7, d.att8, d.att9, d.att10) for page blocks
                data_tabel1(d)
               
                detector_chart1(d.score_lof, d.score_knn, d.score_if, d.score_mahalanobis, d.prediction_if, d.prediction_knn, d.prediction_lof, d.prediction_mahalanobis)            
            }
            else{
                //data_tabel2(d.id, d.att1, d.att2, d.att3, d.att4, d.att5, d.att6, d.att7, d.att8, d.att9, d.att10)
                data_tabel2(d)
                detector_chart2(d.score_lof, d.score_knn, d.score_if, d.score_mahalanobis, d.prediction_if, d.prediction_knn, d.prediction_lof, d.prediction_mahalanobis)

            }
        })


    // function to create pop-up barchart of detectors     
    function detector_chart1(LOF, KNN, IF, Mahalanobis, IF_pred, KNN_pred, LOF_pred, Mahalanobis_pred){
        // store the data
        var data_bar = [
            {
                name: "LOF",
                value: Number(LOF),
                pred: LOF_pred
            },
            {
                name: "KNN",
                value: Number(KNN),
                pred: KNN_pred

            },
            {
                name: "IF",
                value: Number(IF),
                pred: IF_pred
            },
            {
                name: "Mahalanobis",
                value: Number(Mahalanobis),
                pred: Mahalanobis_pred
            }
        ]

        // Check if any detectors were not used. If so remove that name
        for(var i = 0; i<data_bar.length; i++){
            if(data_bar[i]["pred"] == null){
                data_bar.splice(i, 1);
                i--;
            }
        }

        var margin_bar = {top: 30, right: 30, bottom: 70, left: 60},
            width = 360 - margin_bar.left - margin_bar.right,
            height = 300 - margin_bar.top - margin_bar.bottom;
        
        d3.select("#barplot").select("svg").remove(); // reset the plot 

        var svg_bar = d3.select("#barplot")
            .append("svg")
              .attr("width", width + margin_bar.left + margin_bar.right)
              .attr("height", height + margin_bar.top + margin_bar.bottom)
            .append("g")
              .attr("transform",
                    "translate(" + margin_bar.left + "," + margin_bar.top + ")");

        
        // X axis
        var x = d3.scaleBand()
            .range([ 0, width ])
            .domain(data_bar.map((d) => d.name))
            .padding(0.2);
        svg_bar.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x))
            .selectAll("text")
            .attr("transform", "translate(-10,0)rotate(-45)")
            .attr("font-size", "13px")
            .attr("font-family", "arial, sans-serif")
            .style("text-anchor", "end");
        
        // Add Y axis
        var y = d3.scaleLinear()
            .domain([0, Math.max(LOF, KNN, IF, Mahalanobis)*1.1])
            .range([ height, 0])
        svg_bar.append("g")
            .call(d3.axisLeft(y))
        svg_bar.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 0 - margin.left)
            .attr("x",0 - (height / 2))
            .attr("dy", "1em")
            .style("text-anchor", "middle")
            .text("Outlier Score");      

        // add bars
        svg_bar.selectAll("mybar")
            .data(data_bar)
            .enter()
            .append("rect")
                .attr("x", (g) => x(g.name))
                .attr("y", (g) => y(g.value))
                .attr("width", x.bandwidth())
                .attr("height", (g) => height - y(g.value))
                .attr("fill", (g) => color(g.pred))   
    }

    function detector_chart2(LOF, KNN, IF, Mahalanobis, IF_pred, KNN_pred, LOF_pred, Mahalanobis_pred){
        // store the data
        var data_bar = [
            {
                name: "LOF",
                value: Number(LOF),
                pred: LOF_pred
            },
            {
                name: "KNN",
                value: Number(KNN),
                pred: KNN_pred

            },
            {
                name: "IF",
                value: Number(IF),
                pred: IF_pred
            },
            {
                name: "Mahalanobis",
                value: Number(Mahalanobis),
                pred: Mahalanobis_pred
            }
        ]

        // Check if any detectors were not used. If so remove that name
        for(var i = 0; i<data_bar.length; i++){
            if(data_bar[i]["pred"] == null){
                data_bar.splice(i, 1);
                i--;
            }
        }

        var margin_bar = {top: 30, right: 30, bottom: 70, left: 60},
            width = 360 - margin_bar.left - margin_bar.right,
            height = 300 - margin_bar.top - margin_bar.bottom;

        d3.select("#barplot2").select("svg").remove();  // reset the plot 

        var svg_bar = d3.select("#barplot2")
            .append("svg")
              .attr("width", width + margin_bar.left + margin_bar.right)
              .attr("height", height + margin_bar.top + margin_bar.bottom)
            .append("g")
              .attr("transform",
                    "translate(" + margin_bar.left + "," + margin_bar.top + ")");

        
        // X axis
        var x = d3.scaleBand()
            .range([ 0, width ])
            .domain(data_bar.map((d) => d.name))
            .padding(0.2);
        svg_bar.append("g")
            .attr("transform", "translate(0," + height + ")")
            .call(d3.axisBottom(x))
            .selectAll("text")
            .attr("transform", "translate(-10,0)rotate(-45)")
            .attr("font-size", "13px")
            .attr("font-family", "arial, sans-serif")
            .style("text-anchor", "end");
        
        // Add Y axis
        var y = d3.scaleLinear()
            .domain([0, Math.max(LOF, KNN, IF, Mahalanobis)*1.1])
            .range([ height, 0])
        svg_bar.append("g")
            .call(d3.axisLeft(y));
        svg_bar.append("text")
            .attr("transform", "rotate(-90)")
            .attr("y", 0 - margin.left)
            .attr("x",0 - (height / 2))
            .attr("dy", "1em")
            .style("text-anchor", "middle")
            .text("Outlier Score"); 
        
        // add bars
        svg_bar.selectAll("mybar")
            .data(data_bar)
            .enter()
            .append("rect")
                .attr("x", (g) => x(g.name))
                .attr("y", (g) => y(g.value))
                .attr("width", x.bandwidth())
                .attr("height", (g) => height - y(g.value))
                .attr("fill", (g) => color(g.pred))    
    }


    // function to add data tabel                  // for database: we will need to store attributes in a diff tabel: for now hard coded
    function data_tabel1(d){
        var dataSet = []  // stores [col names, value] 
        
        for (let j = 0; j < Object.keys(d).length;j++){
            if (Object.keys(d)[j].startsWith("reliable_") || Object.keys(d)[j].startsWith("score_") || Object.keys(d)[j].startsWith("tsne") || Object.keys(d)[j].startsWith("prediction_") || Object.keys(d)[j] == "correct"|| Object.keys(d)[j] == "prediction" || Object.keys(d)[j] == "label"){
                continue;
            }
            else{
                dataSet.push([Object.keys(d)[j], Object.values(d)[j]])
            }
        }
        
        // var dataSet = [
        //     ["id", id],
        //     ["word_freq_make", att1],
        //     ["word_freq_address", att2],
        //     ["word_freq_all", att3],
        //     ["word_freq_3d", att4],
        //     ["word_freq_our", att5],
        //     ["word_freq_over", att6],
        //     ["word_freq_remove", att7],
        //     ["word_freq_internet", att8],
        //     ["word_freq_order", att9],
        //     ["word_freq_mail", att10],
        // ]
        
        $("#table").DataTable({
            data: dataSet,
            destroy: true,
            searching: false,
            ordering: true,
            paging: false,
            deferRender: true,
            scrollY: 300,
            scroller: true,
            "info": false,
            columns: [
                { title: "Feature"},
                { title: "Value"},
            ]
        });
    }

    function data_tabel2(d){
        var dataSet = []  // stores [col names, value] 
        
        for (let j = 0; j < Object.keys(d).length;j++){
            if (Object.keys(d)[j].startsWith("reliable_") || Object.keys(d)[j].startsWith("score_") || Object.keys(d)[j].startsWith("tsne") || Object.keys(d)[j].startsWith("prediction_") || Object.keys(d)[j] == "correct" || Object.keys(d)[j] == "prediction" || Object.keys(d)[j] == "label"){
                continue;
            }
            else{
                dataSet.push([Object.keys(d)[j], Object.values(d)[j]])
            }
        }
        
        $("#table2").DataTable({
            data: dataSet,
            destroy: true,
            searching: false,
            ordering: true,
            paging: false,
            deferRender: true,
            scrollY: 300,
            scroller: true,
            "info": false,
            columns: [
                { title: "Feature"},
                { title: "Value"},
            ]
        });
    }
   
    // invis rect over chart for zooming
    SVG.append("rect")
    .attr("width", width + 70)
    .attr("height", height + 60)
    .style("fill", "none")
    .style("pointer-events", "all")
    //.attr('transform', 'translate(' + margin.left + ',' + margin.top + ')')
    .call(zoom)
    .call(zoom.transform, transform)  // set starting zoom
    .lower()  // allow mouseover 
    SVG.call(zoom);  // allow zooming even if mouse is on a data point
    
    function updateChart() {  // for zooming
        // recover the new scale
        var newX = d3.event.transform.rescaleX(x);
        var newY = d3.event.transform.rescaleY(y);

        // update axes with these new boundaries
        xAxis.call(d3.axisBottom(newX))
        yAxis.call(d3.axisLeft(newY))

        // update mark position
        scatter
            .selectAll("circle")
            .attr('cx', function(d) {return newX(d.tsne1)})
            .attr('cy', function(d) {return newY(d.tsne2)})            
    }
    
    // following color vars: color points based on fintering inputs
    var color_bool_in = d3.scaleOrdinal()  // color bool vals for multiple 
    .domain([true, false])
    .range(["Steelblue", "#F0F0F0"])

    var color_bool_out = d3.scaleOrdinal()  // color bool vals for multiple 
    .domain([true, false])
    .range(["#FFA500", "#F0F0F0"])

    var correct_bool_pred = d3.scaleOrdinal()  // color bool vals for multiple 
    .domain([true, false])
    .range([function (d){return color(d.prediction)}, "#F0F0F0"])

    var correct_bool_ground = d3.scaleOrdinal()  // color bool vals for multiple 
    .domain([true, false])
    .range([function (d){return color(d.outlier)}, "#F0F0F0"])



    
   // function to change color based on filtering inputs
   // outlier: 1 inlier: 0
   function color_filter(selectedOption_correct, selectedOption_label, metric){
        if(selectedOption_correct == "Both" && selectedOption_label == "Both" && metric == "AutoOD Predictions"){  // just pred
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color(d.prediction)})  // base color
        }
        if(selectedOption_correct == "Both" && selectedOption_label == "Both" && metric == "Ground Truth"){  // just ground truth
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color(d.label)})  // base color
        }
        if(selectedOption_correct == "Both" && selectedOption_label == "Outlier" && metric == "AutoOD Predictions"){  // pred and outlier
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color_bool_out(d.prediction == 1)})
        }
        if(selectedOption_correct == "Both" && selectedOption_label == "Inlier" && metric == "AutoOD Predictions"){  // pred and inlier
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color_bool_in(d.prediction == 0)})
        }
        if(selectedOption_correct == "Both" && selectedOption_label == "Outlier" && metric == "Ground Truth"){  // ground truth and outlier
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color_bool_out(d.label == 1)})
        }
        if(selectedOption_correct == "Both" && selectedOption_label == "Inlier" && metric == "Ground Truth"){  // ground truth and inlier
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color_bool_in(d.label == 0)})
        }
        if(selectedOption_correct == "Yes" && selectedOption_label == "Both" && metric == "AutoOD Predictions"){  // pred and yes
            scatter  // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.prediction)})
            scatter
                .selectAll("circle")
                .style("fill", function (d){return correct_bool_pred(d.correct == 1)})
        }
        if(selectedOption_correct == "No" && selectedOption_label == "Both" && metric == "AutoOD Predictions"){  // pred and no
            scatter // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.prediction)})
            scatter
                .selectAll("circle")
                .style("fill", function (d){return correct_bool_pred(d.correct == 0)})
        }
        if(selectedOption_correct == "Yes" && selectedOption_label == "Both" && metric == "Ground Truth"){  // ground truth and yes
            scatter // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.label)})
            scatter
                .selectAll("circle")
                .style("fill", function (d){return correct_bool_ground(d.correct == 1)})
        }
        if(selectedOption_correct == "No" && selectedOption_label == "Both" && metric == "Ground Truth"){  // ground truth and no
            scatter  // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.label)})
            scatter
                .selectAll("circle")
                .style("fill", function (d){return correct_bool_ground(d.correct == 0)})
        }
        if(selectedOption_correct == "Yes" && selectedOption_label == "Inlier" && metric == "Ground Truth"){  // inlier and ground truth and yes
            scatter  // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.label)})
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color_bool_in(d.correct == 1 && d.label == 0)})
        }
        if(selectedOption_correct == "No" && selectedOption_label == "Inlier" && metric == "Ground Truth"){  // inlier and ground truth and no
            scatter  // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.label)})
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color_bool_in(d.correct == 0 && d.label == 0)})
        }
        if(selectedOption_correct == "Yes" && selectedOption_label == "Outlier" && metric == "Ground Truth"){  // outlier and ground truth and yes
            scatter  // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.label)})
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color_bool_out(d.correct == 1 && d.label == 1)})
        }
        if(selectedOption_correct == "No" && selectedOption_label == "Outlier" && metric == "Ground Truth"){  // outlier and ground truth and no
            scatter  // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.label)})
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color_bool_out(d.correct == 0 && d.label == 1)})
        }
        if(selectedOption_correct == "Yes" && selectedOption_label == "Inlier" && metric == "AutoOD Predictions"){  // inlier and pred and yes
            scatter  // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.prediction)})
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color_bool_in(d.correct == 1 && d.prediction == 0)})
        }
        if(selectedOption_correct == "No" && selectedOption_label == "Inlier" && metric == "AutoOD Predictions"){  // inlier and pred and no
            scatter  // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.prediction)})
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color_bool_in(d.correct == 0 && d.prediction == 0)})
        }
        if(selectedOption_correct == "Yes" && selectedOption_label == "Outlier" && metric == "AutoOD Predictions"){  // outlier and pred and yes
            scatter  // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.prediction)})
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color_bool_out(d.correct == 1 && d.prediction == 1)})
        }
        if(selectedOption_correct == "No" && selectedOption_label == "Outlier" && metric == "AutoOD Predictions"){  // outlier and pred and no
            scatter  // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.prediction)})
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color_bool_out(d.correct == 0 && d.prediction == 1)})
        }
    }

    // starting values
    var selectedOption_correct = "Both"
    var selectedOption_label = "Both"
    var selectedOption_pred = "AutoOD Predictions"

    // filter AutoOD preds
    var options = ["AutoOD Predictions", "Ground Truth"]
    d3.select("#metric")
        .selectAll('myOptions')
            .data(options)
        .enter()
    	    .append('option')
        .text(function (d) {return d;}) // text
        .attr("value", function (d) { return d; }) // value from user input
    
    d3.select("#metric").on("change", function(d) {
        selectedOption_pred = d3.select(this).property("value") // recover user input
        color_filter(selectedOption_correct, selectedOption_label, selectedOption_pred)
    })

    // Filtering correct preds
    var options = ["Both", "Yes", "No"]
    d3.select("#selectButton_correct")
        .selectAll('myOptions')
     	    .data(options)
        .enter()
    	    .append('option')
        .text(function (d) {return d;}) // text
        .attr("value", function (d) { return d; }) // value from user input 

    // button input trigger
    d3.select("#selectButton_correct").on("change", function(d) {
        selectedOption_correct = d3.select(this).property("value") // recover user input
        color_filter(selectedOption_correct, selectedOption_label, selectedOption_pred)
    })

    // Filtering ground truth labels
    var options = ["Both", "Inlier", "Outlier"]
    d3.select("#selectButton_label")
        .selectAll('myOptions')
     	    .data(options)
        .enter()
    	    .append('option')
        .text(function (d) {return d;}) // text
        .attr("value", function (d) { return d; }) // value from user input

    // button input trigger
    d3.select("#selectButton_label").on("change", function(d) {
        selectedOption_label = d3.select(this).property("value") // recover user input
        color_filter(selectedOption_correct, selectedOption_label, selectedOption_pred)
    })

    var color_reliable = d3.scaleOrdinal()  // color for reliable label filter
        .domain([0, 1])
        .range(["#F0F0F0", function (d){return color(d.prediction)}])    
    
        // function to update mark color based on reliable label interation
    function update_reliable(selectedOption) {
        scatter  // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.prediction)})
        if(selectedOption_reliable == "Both"){
            scatter
            .selectAll("circle")
            .style("fill", function (d){return color_reliable(eval("d." + selectedOption))})  // add color to reliable labels DH no
        }
        if(selectedOption_reliable == "Inlier"){
            scatter  // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.prediction)})
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color_bool_in(eval("d." + selectedOption) == 1 && d.prediction == 0)})
        }
        if(selectedOption_reliable == "Outlier"){
            scatter  // reset colors
                .selectAll("circle")
                .style("fill", function (d){return color(d.prediction)})
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color_bool_out(eval("d." + selectedOption) == 1 && d.prediction == 1)})
        }

    }

    // starting value for reliable labels option: reliable inlier or outlier
    var selectedOption_reliable = "Both"
    var options = ["Both", "Inlier", "Outlier"]
    d3.select("#reliable_inout")
        .selectAll('myOptions')
            .data(options)
        .enter()
    	    .append('option')
        .text(function (d) {return d;}) // text
        .attr("value", function (d) { return d; }) // value from user input
    
    d3.select("#reliable_inout").on("change", function(d) {
        selectedOption_reliable = d3.select(this).property("value") // recover user input
        update_reliable(curr_it)
    })

    
    // reliable labels slider
    var round = "round_1"  // global var to keep track of round
    var count_round1 = 0
    var count_round2 = 0
    var curr_it = "reliable_" + (count_round1)

    // get total number of iterations for each round
    for (let i = 0; i < data.length; i++) {
        var colname = String(Object.keys(data[0])[i])
        if(colname.slice(0, 9) == "reliable_"){
            count_round1 = count_round1 + 1
        }
    }

    var sliderStep = d3
        .sliderBottom()
        .min(1)
        .max(count_round1)
        .width(width)
        .ticks(count_round1)
        .step(1)
        .default(count_round1)

    var gStep = d3
        .select('div#slider-step')
        .append('svg')
        .attr('width', 1000)
        .attr('height', 100)
        .append('g')
        .attr('transform', 'translate(30,30)');

    gStep.call(sliderStep);
    
    sliderStep.on('onchange', val => {  // input value on change
        scatter
            .selectAll("circle")
            .style("fill", function (d){return color(d.prediction)})  // reset to normal color
        curr_it ="reliable_" + (val -1);
        document.getElementById("reliable_output").innerHTML = 
        "Percent of reliable labels:" + " " + Math.round(d3.mean(data.map(function (d){return eval("d." + curr_it)})) * 100) + "%";  // average of reliable labels (%);
        update_reliable("reliable_"+ (val -1));
    })

    document.getElementById("reliable_output").innerHTML = 
        "Percent of reliable labels:" + " " + Math.round(d3.mean(data.map(function (d){return eval("d." + curr_it)})) * 100) + "%";  // starting average of reliable labels

    function update_slider(selectedOption) {
        if(selectedOption == "round 1"){
            round = "round_1"
            sliderStep
                .min(1)
                .max(count_round1)
                .width(width)
                .ticks(count_round1)
                .step(1)
                .default(count_round1)
            gStep.call(sliderStep);

        }else if(selectedOption == "round 2"){      
            round = "round_2"
            sliderStep
                .min(1)
                .max(count_round2)
                .width(width)
                .ticks(count_round2)
                .step(1)
                .default(count_round2)
            gStep.call(sliderStep);

        }else{
            scatter
                .selectAll("circle")
                .style("fill", function (d){return color(d.prediction)})  // add normal color
        }

    }


    // button to switch between round 1 and 2
    // var options = [" ", "round 1", "round 2"]
    // d3.select("#selectButton_round")
    //     //.style("position", "absolute")
    //     //.style("left", 50 + "px")
    //     .selectAll('myOptions')
    //  	    .data(options)
    //     .enter()
    // 	    .append('option')
    //     .text(function (d) {return d;}) // text
    //     .attr("value", function (d) { return d; }) // value from user input

    // // button input trigger
    // d3.select("#selectButton_round").on("change", function(d) {
    //     scatter
    //         .selectAll("circle")
    //         .style("fill", function (d){return color(d.pred)})  // reset to normal color
    //     selectedOption_round = d3.select(this).property("value") // recover user input
    //     update_slider(selectedOption_round)
    // })

});

