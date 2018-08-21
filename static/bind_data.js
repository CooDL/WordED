
/* bind data to highcharts pie */
function bind_data_to_pie(divName,data) {
    $(function () {
        $(divName).highcharts({
            chart: {
                plotBackgroundColor: null,
                plotBorderWidth: null,
                plotShadow: false,
            },
            title: {
                text: '抽样检测数'
            },
            credits: {
            	 text: 'optotrace',
            	 href: 'http://www.optotrace.com'
        	 },
             legend: {
                    align: 'right',
                    verticalAlign: 'top',
                     width: 200,
                    itemWidth: 190,
                    layout: 'vertical',
                    labelFormatter: function() {
                return this.name +' (' +this.y+ ')';
            }
                },
           
            plotOptions: {
                pie: {
                    allowPointSelect: true,
                    cursor: 'pointer',
                    dataLabels: {
                        enabled: true,
                        color: '#000000',
                        connectorColor: '#000000',
                        formatter: function() {
                          return this.point.name + ' (' + Highcharts.numberFormat(this.percentage, 2) + '%)';
                       },

                       
                    },
                    showInLegend: true,
                }

            },
            tooltip: {
                formatter: function() {
                    return '<b>' + this.point.name + '</b>: ' + Highcharts.numberFormat(this.percentage, 1) + '% (' + Highcharts.numberFormat(this.y, 0, ',') + ' 个)';
                }
		    },
            series: [{
                type: 'pie',
                name: '检测数：',
                data: data
            }]
        });
    });
}

/* bind data to highchart bar */
function bind_data_to_bar(divName,data,header,height){
$(function () {
        $(divName).highcharts({
            chart: {
                type: 'column',
                inverted: true,
                height:height
            },
            title: {
                text: '抽样检测数'
            },
            credits: {
            	 text: 'optotrace',
            	 href: 'http://www.optotrace.com'
        	 },
            xAxis: {
                categories: header,
                labels: {
                   
                    align: 'right',
                    style: {
                        fontSize: '13px',
                        fontFamily: 'Verdana, sans-serif'
                    }
                }
            },
            yAxis: {
                min: 0,
                title: {
                    text: '样本数目'
                }
            },
            legend: {
                enabled: false
            },
            tooltip: {
                pointFormat: '样本数目: <b>{point.y:.1f} (个)</b>',
            },
            series: [{
                name: '抽样检测数',
                data: data,
                colorByPoint: true,
                dataLabels: {
                    enabled: true,
                    rotation: -90,
                    color: '#FFFFFF',
                    align: 'right',
                    x: 4,
                    y: 10,
                    style: {
                        fontSize: '13px',
                        fontFamily: 'Verdana, sans-serif',
                        textShadow: '0 0 3px black'
                    }
                }
            }]
        });
    });
}

function bind_data_to_line(div_name,categories,tickInterval,data){
$(function () {
            $(div_name).highcharts({
                chart: {
                    zoomType: 'x',
                    spacingRight: 20
                },
                title: {
                    text: ''
                },
              credits: {
            	 text: 'DDOS Forecast System',
            	 href: '#'
        	 },

             xAxis: {
                categories: categories,
                tickInterval: tickInterval,
                gridLineWidth: 1
            },
            yAxis: {
                title: {
                    text: 'Probability'
                },
                plotLines: [{
                    value: 0,
                    width: 1,
                    color: '#808080'
                }]
            },

                tooltip: {
                    shared: true
                },
                legend: {
                    enabled: false
                },
                plotOptions: {
                    line: {
     
                        lineWidth: 2,
                        marker: {
                            enabled: false
                        },
                        shadow: false,
                        states: {
                            hover: {
                                lineWidth: 1
                            }
                        },
                        threshold: null
                    }
                },

           series: [{
                name: 'Probability',
                data: data
            }]
            });
        });
}


function bind_speed(score){
$(function () {
var gaugeOptions = {

        chart: {
            type: 'solidgauge'
        },

        title: null,

        pane: {
            center: ['50%', '85%'],
            size: '140%',
            startAngle: -90,
            endAngle: 90,
            background: {
                backgroundColor: (Highcharts.theme && Highcharts.theme.background2) || '#EEE',
                innerRadius: '60%',
                outerRadius: '100%',
                shape: 'arc'
            }
        },

        tooltip: {
            enabled: false
        },

        // the value axis
        yAxis: {
            stops: [
                [0.1, '#55BF3B'], // green
                [0.5, '#DDDF0D'], // yellow
                [0.9, '#DF5353'] // red
            ],
            lineWidth: 0,
            minorTickInterval: null,
            tickPixelInterval: 400,
            tickWidth: 0,
            title: {
                y: -70
            },
            labels: {
                y: 16
            }
        },

        plotOptions: {
            solidgauge: {
                dataLabels: {
                    y: 5,
                    borderWidth: 0,
                    useHTML: true
                }
            }
        }
    };

    // The speed gauge
    $('#container-speed').highcharts(Highcharts.merge(gaugeOptions, {
        yAxis: {
            min: 0,
            max: 1,
            title: {
                text: 'Normal'
            }
        },

        credits: {
            enabled: false
        },

        series: [{
            name: 'Speed',
            data: [score],
            dataLabels: {
                format: '<div style="text-align:center"><span style="font-size:25px;color:' +
                    ((Highcharts.theme && Highcharts.theme.contrastTextColor) || 'black') + '">{y}</span><br/>' +
                       '<span style="font-size:12px;color:silver">Probability</span></div>'
            },
            tooltip: {
                valueSuffix: ' Probability'
            }
        }]

    }));

});
}