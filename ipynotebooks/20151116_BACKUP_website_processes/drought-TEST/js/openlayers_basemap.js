var arrayOSM;
var arrayAerial;
var baseOSM;
var baseAerial;
var wps, urlWPS, urlRA2, map, layer;
var lattop, latbottom, longleft, longright;
var minlon, minlat, maxlon, maxlat;
var urlWPS = "../../cgi-bin/pywps.cgi";
var vectors;
var box;
var transform;
var markers;
var fromProjection = new OpenLayers.Projection("EPSG:4326"); // transform from WGS 1984
var toProjection = new OpenLayers.Projection("EPSG:900913"); // to Spherical Mercator Projection


function pad(n) {
    return (n < 10) ? ("0" + n) : n;
}

function endsWith(str, suffix) {
    return str.indexOf(suffix, str.length - suffix.length) !== -1;
}

function endDrag_maps(bbox) {
    if ($('#bbox_result_order').length > 0) {
        // dostuff
        endDrag_order(bbox)
    };
    if ($('#bbox_result_maps').length > 0) {
        // dostuff
        var bounds = bbox.getBounds();
        setBounds_maps(bounds);
        if ($('#bbox_result_order').length == 0) {
            drawBox(bounds)
        };
        box.deactivate();
        document.getElementById("bbox_drag_instruction_maps").style.display = 'none';
        document.getElementById("bbox_adjust_instruction_maps").style.display = 'block';
        document.getElementById("bbox_drag_starter_maps").style.display = 'none';
    };
}

function endDrag_order(bbox) {
    var bounds = bbox.getBounds();
    setBounds_order(bounds);
    drawBox(bounds);
    box.deactivate();
    document.getElementById("bbox_drag_instruction_order").style.display = 'none';
    document.getElementById("bbox_adjust_instruction_order").style.display = 'block';
    document.getElementById("bbox_drag_starter_order").style.display = 'none';
}

function dragNewBox_maps() {
    box.activate();
    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    document.getElementById("bbox_drag_instruction_maps").style.display = 'block';
    document.getElementById("bbox_adjust_instruction_maps").style.display = 'none';
    document.getElementById("bbox_drag_starter_maps").style.display = 'none';
    setBounds_maps(null);
}

function dragNewBox_order() {
    box.activate();
    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    document.getElementById("bbox_drag_instruction_order").style.display = 'block';
    document.getElementById("bbox_adjust_instruction_order").style.display = 'none';
    document.getElementById("bbox_drag_starter_order").style.display = 'none';
    setBounds_order(null);
}

function removeBox_maps() {
    box.deactivate();
    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    document.getElementById("bbox_drag_instruction_maps").style.display = 'none';
    document.getElementById("bbox_adjust_instruction_maps").style.display = 'none';
    document.getElementById("bbox_drag_starter_maps").style.display = 'block';
    setBounds_maps(null);
}

function removeBox_order() {
    box.deactivate();
    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    document.getElementById("bbox_drag_instruction_order").style.display = 'none';
    document.getElementById("bbox_adjust_instruction_order").style.display = 'none';
    document.getElementById("bbox_drag_starter_order").style.display = 'block';
    setBounds_order(null);
}

function boxResize_maps(event) {
    if ($('#bbox_result_order').length > 0) {
        // dostuff
        boxResize_order(event)
    };
    if ($('#bbox_result_maps').length > 0) {
        // dostuff
        setBounds_maps(event.feature.geometry.bounds);
    };
}

function boxResize_order(event) {
    setBounds_order(event.feature.geometry.bounds);
}

function drawBox(bounds) {
    var feature = new OpenLayers.Feature.Vector(bounds.toGeometry());
    vectors.addFeatures(feature);
    transform.setFeature(feature);
}

function toPrecision(zoom, value) {
    var decimals = Math.pow(10, Math.floor(zoom / 3));
    return Math.round(value * decimals) / decimals;
}

function setBounds_maps(bounds) {
    if (bounds == null) {
        document.getElementById("bbox_result_maps").innerHTML = "";
    } else {
        b = bounds.clone().transform(map.getProjectionObject(), new OpenLayers.Projection(
            "EPSG:4326"))
        minlon = toPrecision(map.getZoom(), b.left);
        minlat = toPrecision(map.getZoom(), b.bottom);
        maxlon = toPrecision(map.getZoom(), b.right);
        maxlat = toPrecision(map.getZoom(), b.top);
        document.getElementById("bbox_result_maps").innerHTML = "<table>" + "<tr>" + "<td>" +
            "minlon=" + minlon + "</td>" + "<td>" + "minlat=" + minlat + "</td>" + "</tr>" +
            "<tr>" + "<td>" + "maxlon=" + maxlon + "</td>" + "<td>" + "maxlat=" + maxlat +
            "</td>" + "</tr>" + "</table>";
	//b = [b[Object.keys(b)[0]],b[Object.keys(b)[1]],b[Object.keys(b)[2]],b[Object.keys(b)[3]]];
	c = [owl.deepCopy(minlon),owl.deepCopy(minlat),owl.deepCopy(maxlon),owl.deepCopy(maxlat)];
    }
}

function setBounds_order(bounds) {
        if (bounds == null) {
            document.getElementById("bbox_result_order").innerHTML = "";
        } else {
            b = bounds.clone().transform(map.getProjectionObject(), new OpenLayers.Projection(
                "EPSG:4326"))
            minlon = toPrecision(map.getZoom(), b.left);
            minlat = toPrecision(map.getZoom(), b.bottom);
            maxlon = toPrecision(map.getZoom(), b.right);
            maxlat = toPrecision(map.getZoom(), b.top);
            document.getElementById("bbox_result_order").innerHTML = "<table>" + "<tr>" + "<td>" +
                "minlon=" + minlon + "</td>" + "<td>" + "minlat=" + minlat + "</td>" + "</tr>" +
                "<tr>" + "<td>" + "maxlon=" + maxlon + "</td>" + "<td>" + "maxlat=" + maxlat +
                "</td>" + "</tr>" + "</table>";
            //b = [b[Object.keys(b)[0]],b[Object.keys(b)[1]],b[Object.keys(b)[2]],b[Object.keys(b)[3]]];
            c = [owl.deepCopy(minlon),owl.deepCopy(minlat),owl.deepCopy(maxlon),owl.deepCopy(maxlat)];
        }
    }
    // --- // --- // --- // --- // --- // --- // --- // --- //


// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// Execute Metadata Internal Query through PyWPS
function executeMetadataInternalQuery() {
    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecutedMetadataInternal
    });


    // define the process and append it to OpenLayers.WPS instance
    var Metadata_Process_Internal = new OpenLayers.WPS.Process({
        identifier: "WPS_METADATA_INTERNAL",
        outputs: [
            new OpenLayers.WPS.LiteralPut({
                identifier: "LST_MOD11C2005",
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "NDVI_MOD13C1005",
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "NDAI_1km",
            }) 
        ]
    });


    // defined earlier
    wps.addProcess(Metadata_Process_Internal);

    // run Execute
    wps.execute("WPS_METADATA_INTERNAL");      

};

/**
 * WPS events
 */
// Everything went OK 
function onExecutedMetadataInternal(process) {
    // LST_MOD11C2005    
    // remove first and last character, remove the quotes, remove unnecesary spaces and split csv
    var LST_MOD11C2005 = process.outputs[0].getValue().slice(1,-1).replace(/['"]+/g, '').replace(/\s+/g, '').split(',');
    window.metadata_LST_MOD11C2005_from_date = LST_MOD11C2005[0];
    window.metadata_LST_MOD11C2005_to_date = LST_MOD11C2005[1];
    window.metadata_LST_MOD11C2005_temp_res = LST_MOD11C2005[2];
    window.metadata_LST_MOD11C2005_lonmin = LST_MOD11C2005[3];
    window.metadata_LST_MOD11C2005_latmin = LST_MOD11C2005[4];
    window.metadata_LST_MOD11C2005_lonmax = LST_MOD11C2005[5];
    window.metadata_LST_MOD11C2005_latmax = LST_MOD11C2005[6];
    window.metadata_LST_MOD11C2005_spat_res = LST_MOD11C2005[7];

    // NDVI_MOD13C1005
    var NDVI_MOD13C1005 = process.outputs[1].getValue().slice(1,-1).replace(/['"]+/g, '').replace(/\s+/g, '').split(',');
    window.metadata_NDVI_MOD13C1005_from_date = NDVI_MOD13C1005[0];
    window.metadata_NDVI_MOD13C1005_to_date = NDVI_MOD13C1005[1];
    window.metadata_NDVI_MOD13C1005_temp_res = NDVI_MOD13C1005[2];
    window.metadata_NDVI_MOD13C1005_lonmin = NDVI_MOD13C1005[3];
    window.metadata_NDVI_MOD13C1005_latmin = NDVI_MOD13C1005[4];
    window.metadata_NDVI_MOD13C1005_lonmax = NDVI_MOD13C1005[5];
    window.metadata_NDVI_MOD13C1005_latmax = NDVI_MOD13C1005[6];
    window.metadata_NDVI_MOD13C1005_spat_res = NDVI_MOD13C1005[7];

    // NDAI_1km
    var NDAI_1km = process.outputs[2].getValue().slice(1,-1).replace(/['"]+/g, '').replace(/\s+/g, '').split(',');
    window.metadata_NDAI_1km_from_date = NDAI_1km[0];
    window.metadata_NDAI_1km_to_date = NDAI_1km[1];
    window.metadata_NDAI_1km_temp_res = NDAI_1km[2];
    window.metadata_NDAI_1km_lonmin = NDAI_1km[3];
    window.metadata_NDAI_1km_latmin = NDAI_1km[4];
    window.metadata_NDAI_1km_lonmax = NDAI_1km[5];
    window.metadata_NDAI_1km_latmax = NDAI_1km[6];
    window.metadata_NDAI_1km_spat_res = NDAI_1km[7];

    console.log('function Metadata Internal works');
};

// END Execute Metadata Internal Query
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// Execute Metadata External Query through PyWPS
function executeMetadataExternalQuery() {
    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecutedMetadataExternal
    });


    // define the process and append it to OpenLayers.WPS instance
    var Metadata_Process_External = new OpenLayers.WPS.Process({
        identifier: "WPS_METADATA_EXTERNAL",
        outputs: [
            new OpenLayers.WPS.LiteralPut({
                identifier: "gpcp",
            })
        ]
    });


    // defined earlier
    wps.addProcess(Metadata_Process_External);

    // run Execute
    wps.execute("WPS_METADATA_EXTERNAL");      

};

/**
 * WPS events
 */
// Everything went OK 
function onExecutedMetadataExternal(process) {
    // gpcp
    var gpcp = process.outputs[0].getValue().slice(1,-1).replace(/['"]+/g, '').replace(/\s+/g, '').split(',');
    window.metadata_gpcp_from_date = gpcp[0];
    window.metadata_gpcp_to_date = gpcp[1];
    window.metadata_gpcp_temp_res = gpcp[2];
    window.metadata_gpcp_lonmin = gpcp[3];
    window.metadata_gpcp_latmin = gpcp[4];
    window.metadata_gpcp_lonmax = gpcp[5];
    window.metadata_gpcp_latmax = gpcp[6];
    window.metadata_gpcp_spat_res = gpcp[7];

    console.log('function Metadata External works');
};

// END Execute Metadata External Query
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------

var Init; (Init = function Init () {
    console.log( "ready!" );
    executeMetadataInternalQuery();
    executeMetadataExternalQuery()
})();

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// Execute executeBoxCounty Query through PyWPS
function executeBoxCounty() {



    document.getElementById("county_stats_gobox_refresh").style.display = 'block';

// check if county is selected, otherwise state dialog error
if (document.getElementById('county').value == 'County') {    

    var $message_bbox_dropdown_china = $('<div></div>')
        .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No county selected, make sure to select a province, region and county')
        .dialog({
        dialogClass: "no-close",
        modal: true,
        resizable: false,
        title: 'Warning',
        buttons: {
            OK: function () {	    
                $(this).dialog("close");
		$message_bbox_dropdown_china.remove();
                document.getElementById("county_stats_gobox_refresh").style.display = 'none';
            }
        }
    });

    $message_bbox_dropdown_china.dialog('open').parent().addClass("ui-state-error");

    return
} else {
    // county is selected
    console.log('selected county:' + document.getElementById('county').value);
};


    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecutedBoxCounty
    });

    // definte the process and append it to OpenLayers.WPS instance
    var BoxCounty_Process = new OpenLayers.WPS.Process({
        identifier: "china_geojson",
        inputs: [
           new OpenLayers.WPS.LiteralPut({
               identifier: "Province",
               value: document.getElementById('province').value,//'Anhui',//get value from ID from select 
               //value: document.getElementById("pyhants_yes_no_nvai").value,
 
           }),
           new OpenLayers.WPS.LiteralPut({
               identifier: "Prefecture",
               value: document.getElementById('prefecture').value,//'Bengbu',//get value from ID from select
               //value: document.getElementById("pyhants_yes_no_nvai").value,

           }),
           new OpenLayers.WPS.LiteralPut({
               identifier: "County",
               value: document.getElementById('county').value,//get value from ID from select
               //value: document.getElementById("pyhants_yes_no_nvai").value,

           })
        ],
        async: false,
        outputs: [
            new OpenLayers.WPS.ComplexPut({
                identifier: "Bound_GeoJSON",
                asReference: true
            })
        ]

    });


    // defined earlier
    wps.addProcess(BoxCounty_Process);

    // run Execute
    wps.execute("china_geojson");

};

//define variable that can store the extent of the geojson vector layer
var geojson_bounds;
// Everything went OK
function onExecutedBoxCounty(process) {
    console.log('get some output')
    // MODIS_13_C1_COV
    geojson_out = process.outputs[0].getValue();

    console.log(geojson_out); // "http://localhost/wps/wpsoutputs/Bound_GeoJSON-10373308-7d00-11e5-a2e8-4437e647de9f.geojson"
    geojson_out_slice = geojson_out.slice(17, 90);
    console.log('geojson_out_slice: ' + geojson_out_slice);
    geojson_link = '../../',
    geojson_link += geojson_out_slice;
    //image_link_pap_saved = JSON.stringify(image_link);
    //"http://localhost/cgi-bin/mapserv?map=/var/www/html/wpsoutputs/wps23332-tmpKCoWpM.map"
    console.log('geojson_link: ' + geojson_link);


    geojson_layer = new OpenLayers.Layer.Vector("GeoJSON", {
        projection: new OpenLayers.Projection('EPSG:900931'),
        strategies: [new OpenLayers.Strategy.Fixed()],
        protocol: new OpenLayers.Protocol.HTTP({
            url: geojson_link,
            format: new OpenLayers.Format.GeoJSON()
        })
    });
    map.addLayer(geojson_layer);
    // zoom to  dataextent
    geojson_layer.events.register('loadend', geojson_layer, function(evt){map.zoomToExtent(geojson_layer.getDataExtent() )} );
    //console.log(geojson_layer.getDataExtent());
    // store data extent for future WCS request (over WPS)
    var proj4326 = new OpenLayers.Projection("EPSG:4326"); // transform from WGS 1984
    var proj900913 = new OpenLayers.Projection("EPSG:900913"); // to Spherical Mercator Projection
    geojson_layer.events.register('loadend', geojson_layer, function(evt){
        geojson_bounds = geojson_layer.getDataExtent().transform(proj900913, proj4326), 
        console.log(geojson_bounds) 
    } );
    document.getElementById("county_stats_gobox_refresh").style.display = 'none';

};



// END executeBoxCounty Query
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------


function executeGetExentCounty() {alert(geojson_bounds)};

var histogram_chart;
function loadScript(url, callback)
{
    // Adding the script tag to the head as suggested before
    var head = document.getElementsByTagName('head')[0];
    var script = document.createElement('script');
    script.type = 'text/javascript';
    script.src = url;

    // Then bind the event to the callback function.
    // There are several events for cross browser compatibility.
    script.onreadystatechange = callback;
    script.onload = callback;

    // Fire the loading
    head.appendChild(script);
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// Execute executeHistogramComputationQuery() through PyWPS
function executeHistogramComputationQuery() {

document.getElementById("county_stats_hist_refresh").style.display = 'block';
    
// check if county is selected, otherwise state dialog error
if (document.getElementById('county').value == 'County') {    

    var $message_bbox_dropdown_china = $('<div></div>')
        .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No county selected, make sure to select a province, region and county')
        .dialog({
        dialogClass: "no-close",
        modal: true,
        resizable: false,
        title: 'Warning',
        buttons: {
            OK: function () {	    
                $(this).dialog("close");
		$message_bbox_dropdown_china.remove();
                document.getElementById("county_stats_hist_refresh").style.display = 'none';
            }
        }
    });

    $message_bbox_dropdown_china.dialog('open').parent().addClass("ui-state-error");

    return
} else {
    // county is selected
    console.log('selected county:' + document.getElementById('county').value);
};

// check if county geojson boundaries are known
if (typeof geojson_bounds === 'undefined') {
    // your code here.
    var $message_geojson_bounds = $('<div></div>')
        .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No known boundary extent of county, first click "GO" after county selection is made. Lets do it know and try again')
        .dialog({
        dialogClass: "no-close",
        modal: true,
        resizable: false,
        title: 'Warning',
        buttons: {
            OK: function () {
                $(this).dialog("close");
                $message_geojson_bounds.remove();
                executeBoxCounty();
                document.getElementById("county_stats_hist_refresh").style.display = 'none';
            }
        }
    });

    $message_geojson_bounds.dialog('open').parent().addClass("ui-state-error");

    return
} else {
    // county extent is known
    longleft = geojson_bounds.left;
    longright = geojson_bounds.right;
    lattop = geojson_bounds.top;
    latbottom = geojson_bounds.bottom;

    var bbox_geojson = [longleft, latbottom, longright, lattop];
    console.log('county extent:' + bbox_geojson);
};

// if date is not openend, set default date
if ($('#startDate_map').length > 0) {
// dostuff
date_in = $('#startDate_map').MonthPicker('GetSelectedYear') + '-' + pad($(
    '#startDate_map').MonthPicker('GetSelectedMonth')) + '-' + '01';
} else {
date_in = '2014-08-01';
console.log('date_in not undefined, set to default: 2014-08-01');
};



// check if dataset is available based on metadata
if (typeof metadata_NDAI_1km_to_date !== 'undefined') {
    // metadata_NDAI_1km_to_date is defined
    // check if selected date is AFTER available date
    if (date_in > metadata_NDAI_1km_to_date) {
        // your code here.
        var $message_metadata_NDAI_1km = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The selected date is AFTER the available date! NDAI dataset availability: ' + metadata_NDAI_1km_from_date + ' to ' + metadata_NDAI_1km_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_NDAI_1km.remove();
                    document.getElementById("county_stats_hist_refresh").style.display = 'none';
                    //$('#map_di_pap_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_NDAI_1km.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    // check if selected date is BEFORE available date
    if (date_in < metadata_NDAI_1km_from_date) {
        // your code here.
        var $message_metadata_NDAI_1km = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The selected date is BEFORE the available date! NDAI dataset availability: ' + metadata_NDAI_1km_from_date + ' to ' + metadata_NDAI_1km_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_NDAI_1km.remove();
                    document.getElementById("county_stats_hist_refresh").style.display = 'none';
                    //$('#map_di_pap_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_NDAI_1km.dialog('open').parent().addClass("ui-state-error");
        return;
    };

    run_the_Histogram_Query();

} else {
    // option include cancel or continue
    console.log('aaaw');
    function confirmation() { 
	console.log('checking...');
	var defer = $.Deferred();

	var $message_NO_metadata_NDAI_1km = $('<div></div>')
	    .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No metadata available, click OK to continue or cancel to go back to safety')
	    .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
	    title: 'Warning',
	    buttons: {
	        Continue: function () {
	            defer.resolve("true");
	    	    $(this).dialog("close");
		    run_the_Histogram_Query();
            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close");
                    document.getElementById("county_stats_hist_refresh").style.display = 'none';
                    //$('#map_di_pap_refresh').attr('src', './js/img/refresh.png');
     	        },
	    },
	    close: function () {
		$(this).remove();
            }
	});

	$message_NO_metadata_NDAI_1km.dialog('open').parent().addClass("ui-state-error");
	return defer.promise()

    };


    confirmation().then(function (answer) {
	console.log(answer);
	var ansbool = answer == "true";
	if (ansbool) {
	    console.log('continue as answer confirmation is ' + ansbool); //TRUE
	} else {
	    console.log('return as answer confirmation is ' + ansbool); //FALSE
	    return;
	}
    });    
};






function run_the_Histogram_Query() {
    modal_chart_dialog.dialog("open");//$(".modal_chart").dialog("open");
	//Display right chart
	document.getElementById("drought-derived-chart").style.display = 'none';
	document.getElementById("hydrology-chart").style.display = 'none';
	document.getElementById("vegetation-chart").style.display = 'none';
	document.getElementById("meteorology-chart").style.display = 'none';
	document.getElementById("histogram-chart").style.display = 'block';
	document.getElementById("fracarea-chart").style.display = 'none';

    var date_in_ = JSON.stringify(date_in);
    console.log('date_in: ' + date_in_);
    console.log('date_in_saved_pap: ' + date_in_saved_pap);

    histogram_chart.showLoading();
    console.log('loading started histogram chart');

    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecutedHistogramComputation
    });

    // definte the process and append it to OpenLayers.WPS instance
    var HistogramComputation_Process = new OpenLayers.WPS.Process({
        identifier: "WPS_HISTOGRAM_COMPUTATION",
        inputs: [
           new OpenLayers.WPS.LiteralPut({
               identifier: "Province",
               value: document.getElementById('province').value,//'Anhui',//get value from ID from select
           }),
           new OpenLayers.WPS.LiteralPut({
               identifier: "Prefecture",
               value: document.getElementById('prefecture').value,//'Bengbu',//get value from ID from select
           }),
           new OpenLayers.WPS.LiteralPut({
               identifier: "County",
               value: document.getElementById('county').value,//'Guzhen',//get value from ID from select
           }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "ExtentCounty",
                value: bbox_geojson,
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "date",
                value: date_in,
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "num_observations",
                value: 4,//document.getElementById("histogram_observations").value,
	    }), 
        ],
        async: false,
        outputs: [
            new OpenLayers.WPS.LiteralPut({identifier: "label_ts1"}),
            new OpenLayers.WPS.ComplexPut({
                identifier: "hist_ts1",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),

            new OpenLayers.WPS.LiteralPut({identifier: "label_ts2"}),
            new OpenLayers.WPS.ComplexPut({
                identifier: "hist_ts2",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),

            new OpenLayers.WPS.LiteralPut({identifier: "label_ts3"}),
            new OpenLayers.WPS.ComplexPut({
                identifier: "hist_ts3",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),

            new OpenLayers.WPS.LiteralPut({identifier: "label_ts4"}),
            new OpenLayers.WPS.ComplexPut({
                identifier: "hist_ts4",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),

            new OpenLayers.WPS.LiteralPut({identifier: "label_ts5"}),
            new OpenLayers.WPS.ComplexPut({
                identifier: "hist_ts5",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),
        ]
    });


    // defined earlier
    wps.addProcess(HistogramComputation_Process);

    // run Execute
    wps.execute("WPS_HISTOGRAM_COMPUTATION");

    var cache = [];
    console.log(JSON.stringify(HistogramComputation_Process, function(key, value) {
        if (typeof value === 'object' && value !== null) {
            if (cache.indexOf(value) !== -1) {
                // Circular reference found, discard key
                return;
            }
            // Store value in our collection
            cache.push(value);
        }
        return value;
    }));
    cache = null; // Enable garbage collection

};
};

/**

 * WPS events
 */
// Everything went OK

function onExecutedHistogramComputation(process) {
    console.log('we are within the onExecutedHistogramComputation function:)')
    var label_ts1 = process.outputs[0].getValue()

    var result1 = process.outputs[1].getValue()
    resu1 = result1.slice(16);
    series_ts1 = '..'
    series_ts1 += resu1 //series1

    var label_ts2 = process.outputs[2].getValue()

    var result2 = process.outputs[3].getValue()
    resu2 = result2.slice(16);
    series_ts2 = '..'
    series_ts2 += resu2

    var label_ts3 = process.outputs[4].getValue()

    var result3 = process.outputs[5].getValue()
    resu3 = result3.slice(16);
    series_ts3 = '..'
    series_ts3 += resu3

    var label_ts4 = process.outputs[6].getValue()

    var result4 = process.outputs[7].getValue()
    resu4 = result4.slice(16);
    series_ts4 = '..'
    series_ts4 += resu4

    var label_ts5 = process.outputs[8].getValue()

    var result5 = process.outputs[9].getValue()
    resu5 = result5.slice(16);
    series_ts5 = '..'
    series_ts5 += resu5


    //$('#result_url1').text(result1);
    //$('#result_url2').text(result2);
    console.log('function HistogramComputation works: '+series_ts1);
    console.log('function HistogramComputation works: '+label_ts1);

    //OpenLayers.Util.getElement("result").innerHTML = resu1;


    //var chart = $('#chart1').highcharts();

    $.getJSON(series_ts1, function(data_ts) {
        histogram_chart.series[0].update({
            name: label_ts1,
            data: data_ts
        })
    });
    $.getJSON(series_ts2, function(data_ts) {
        histogram_chart.series[1].update({
            name: label_ts2,        
            data: data_ts
        })
    });
    $.getJSON(series_ts3, function(data_ts) {
        histogram_chart.series[2].update({
            name: label_ts3,
            data: data_ts
        })
    });
    $.getJSON(series_ts4, function(data_ts) {
        histogram_chart.series[3].update({
            name: label_ts4,
            data: data_ts
        })
    });
    $.getJSON(series_ts5, function(data_ts) {
        histogram_chart.series[4].update({
            name: label_ts5,
            data: data_ts
        })
    });

    // refresh button finished loading
    //$('#ts_drought_refresh').attr('src', './js/img/refresh.png');
    histogram_chart.hideLoading();
    histogram_chart.reflow();
    document.getElementById("county_stats_hist_refresh").style.display = 'none';


};

// ------- //



var fracarea_chart;
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// Execute executeFracAreaQuery() through PyWPS
function executeFracAreaQuery() {
                document.getElementById("county_stats_frac_refresh").style.display = 'block';
    
// check if county is selected, otherwise state dialog error
if (document.getElementById('county').value == 'County') {    

    var $message_bbox_dropdown_china = $('<div></div>')
        .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No county selected, make sure to select a province, region and county')
        .dialog({
        dialogClass: "no-close",
        modal: true,
        resizable: false,
        title: 'Warning',
        buttons: {
            OK: function () {	    
                $(this).dialog("close");
		$message_bbox_dropdown_china.remove();
                document.getElementById("county_stats_frac_refresh").style.display = 'none';
            }
        }
    });

    $message_bbox_dropdown_china.dialog('open').parent().addClass("ui-state-error");

    return
} else {
    // county is selected
    console.log('selected county:' + document.getElementById('county').value);
};

// check if county geojson boundaries are known
if (typeof geojson_bounds === 'undefined') {
    // your code here.
    var $message_geojson_bounds = $('<div></div>')
        .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No known boundary extent of county, first click "GO" after county selection is made')
        .dialog({
        dialogClass: "no-close",
        modal: true,
        resizable: false,
        title: 'Warning',
        buttons: {
            OK: function () {
                $(this).dialog("close");
                $message_geojson_bounds.remove();
                document.getElementById("county_stats_frac_refresh").style.display = 'none';
            }
        }
    });

    $message_geojson_bounds.dialog('open').parent().addClass("ui-state-error");

    return
} else {
    // county extent is known
    longleft = geojson_bounds.left;
    longright = geojson_bounds.right;
    lattop = geojson_bounds.top;
    latbottom = geojson_bounds.bottom;

    var bbox_geojson = [longleft, latbottom, longright, lattop];
    console.log('county extent:' + bbox_geojson);
};

// if date is not openend, set default date
if ($('#startDate_map').length > 0) {
// dostuff
date_in = $('#startDate_map').MonthPicker('GetSelectedYear') + '-' + pad($(
    '#startDate_map').MonthPicker('GetSelectedMonth')) + '-' + '01';
} else {
date_in = '2014-08-01';
console.log('date_in not undefined, set to default: 2014-08-01');
};


// check if dataset is available based on metadata
if (typeof metadata_NDAI_1km_to_date !== 'undefined') {
    // metadata_gpcp_to_date is defined
    // check if selected date is AFTER available date
    if (date_in > metadata_NDAI_1km_to_date) {
        // your code here.
        var $message_metadata_NDAI_1km = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The selected date is AFTER the available date! NDAI dataset availability: ' + metadata_NDAI_1km_from_date + ' to ' + metadata_NDAI_1km_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_NDAI_1km.remove();
                    document.getElementById("county_stats_frac_refresh").style.display = 'none';
                }
            }
        });
        $message_metadata_NDAI_1km.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    // check if selected date is BEFORE available date
    if (date_in < metadata_NDAI_1km_from_date) {
        // your code here.
        var $message_metadata_NDAI_1km = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The selected date is BEFORE the available date! NDAI dataset availability: ' + metadata_NDAI_1km_from_date + ' to ' + metadata_NDAI_1km_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_NDAI_1km.remove();
                    document.getElementById("county_stats_frac_refresh").style.display = 'none';
                }
            }
        });
        $message_metadata_NDAI_1km.dialog('open').parent().addClass("ui-state-error");
        return;
    };


    run_the_FracQuery();

} else {
    // option including cancel or continue
    console.log('aaaw');
    function confirmation() { 
	console.log('checking...');
	var defer = $.Deferred();

	var $message_NO_metadata_NDAI_1km = $('<div></div>')
	    .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No metadata available, click OK to continue or cancel to go back to safety')
	    .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
	    title: 'Warning',
	    buttons: {
	        Continue: function () {
	            defer.resolve("true");
	    	    $(this).dialog("close");
		    run_the_FracQuery();
            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close");
                    document.getElementById("county_stats_frac_refresh").style.display = 'none';
     	        },
	    },
	    close: function () {
		$(this).remove();
            }
	});

	$message_NO_metadata_NDAI_1km.dialog('open').parent().addClass("ui-state-error");
	return defer.promise()

    };


    confirmation().then(function (answer) {
	console.log(answer);
	var ansbool = answer == "true";
	if (ansbool) {
	    console.log('continue as answer confirmation is ' + ansbool); //TRUE
	} else {
	    console.log('return as answer confirmation is ' + ansbool); //FALSE
	    return;
	}
    });


    //alert("Metadata is not defined, we will continue anyway. May the Force be with you");
};

function run_the_FracQuery() {
    modal_chart_dialog.dialog("open");//$(".modal_chart").dialog("open");
	//Display right chart
	document.getElementById("drought-derived-chart").style.display = 'none';
	document.getElementById("hydrology-chart").style.display = 'none';
	document.getElementById("vegetation-chart").style.display = 'none';
	document.getElementById("meteorology-chart").style.display = 'none';
	document.getElementById("histogram-chart").style.display = 'none';
	document.getElementById("fracarea-chart").style.display = 'block';

    var date_in_ = JSON.stringify(date_in);
    console.log('date_in: ' + date_in_);
    console.log('date_in_saved_pap: ' + date_in_saved_pap);

    //histogram_chart.showLoading();
    fracarea_chart.showLoading();
    fracarea_chart.reflow();
    console.log('loading started fracarea chart');






    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecutedFracareaComputation
    });

    // definte the process and append it to OpenLayers.WPS instance
    var FracareaComputation_Process = new OpenLayers.WPS.Process({
        identifier: "WPS_FRACAREA_COMPUTATION",
        inputs: [
           new OpenLayers.WPS.LiteralPut({
               identifier: "Province",
               value: document.getElementById('province').value,//'Anhui',//get value from ID from select
           }),
           new OpenLayers.WPS.LiteralPut({
               identifier: "Prefecture",
               value: document.getElementById('prefecture').value,//'Bengbu',//get value from ID from select
           }),
           new OpenLayers.WPS.LiteralPut({
               identifier: "County",
               value: document.getElementById('county').value,//'Guzhen',//get value from ID from select
           }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "ExtentCounty",
                value: bbox_geojson,
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "date",
                value: date_in,
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "num_observations",
                value: 4,//document.getElementById("histogram_observations").value,
	    }), 
        ],
        async: false,
        outputs: [
            new OpenLayers.WPS.LiteralPut({identifier: "label_ts1"}),
            new OpenLayers.WPS.ComplexPut({
                identifier: "frac_ts1",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),

            new OpenLayers.WPS.LiteralPut({identifier: "label_ts2"}),
            new OpenLayers.WPS.ComplexPut({
                identifier: "frac_ts2",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),

            new OpenLayers.WPS.LiteralPut({identifier: "label_ts3"}),
            new OpenLayers.WPS.ComplexPut({
                identifier: "frac_ts3",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),

            new OpenLayers.WPS.LiteralPut({identifier: "label_ts4"}),
            new OpenLayers.WPS.ComplexPut({
                identifier: "frac_ts4",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),

            new OpenLayers.WPS.LiteralPut({identifier: "label_ts5"}),
            new OpenLayers.WPS.ComplexPut({
                identifier: "frac_ts5",
                asReference: true
                    //formats:{mimeType:'application/json'}
            }),
        ]
    });


    // defined earlier
    wps.addProcess(FracareaComputation_Process);

    // run Execute
    wps.execute("WPS_FRACAREA_COMPUTATION");

    var cache = [];
    console.log(JSON.stringify(FracareaComputation_Process, function(key, value) {
        if (typeof value === 'object' && value !== null) {
            if (cache.indexOf(value) !== -1) {
                // Circular reference found, discard key
                return;
            }
            // Store value in our collection
            cache.push(value);
        }
        return value;
    }));
    cache = null; // Enable garbage collection

};
};

/**

 * WPS events
 */
// Everything went OK

function onExecutedFracareaComputation(process) {
    console.log('we are within the onExecutedHistogramComputation function:)')
    var label_ts1 = process.outputs[0].getValue()

    var result1 = process.outputs[1].getValue()
    resu1 = result1.slice(16);
    series_ts1 = '..'
    series_ts1 += resu1 //series1

    var label_ts2 = process.outputs[2].getValue()

    var result2 = process.outputs[3].getValue()
    resu2 = result2.slice(16);
    series_ts2 = '..'
    series_ts2 += resu2

    var label_ts3 = process.outputs[4].getValue()

    var result3 = process.outputs[5].getValue()
    resu3 = result3.slice(16);
    series_ts3 = '..'
    series_ts3 += resu3

    var label_ts4 = process.outputs[6].getValue()

    var result4 = process.outputs[7].getValue()
    resu4 = result4.slice(16);
    series_ts4 = '..'
    series_ts4 += resu4

    var label_ts5 = process.outputs[8].getValue()

    var result5 = process.outputs[9].getValue()
    resu5 = result5.slice(16);
    series_ts5 = '..'
    series_ts5 += resu5


    //$('#result_url1').text(result1);
    //$('#result_url2').text(result2);
    console.log('function FracareaComputation works: '+series_ts1);
    console.log('function FracareaComputation works: '+label_ts1);

    //OpenLayers.Util.getElement("result").innerHTML = resu1;


    //var chart = $('#chart1').highcharts();

    $.getJSON(series_ts1, function(data_ts) {
        fracarea_chart.series[0].update({
            name: label_ts1,
            data: data_ts
        })
    });
    $.getJSON(series_ts2, function(data_ts) {
        fracarea_chart.series[1].update({
            name: label_ts2,        
            data: data_ts
        })
    });
    $.getJSON(series_ts3, function(data_ts) {
        fracarea_chart.series[2].update({

            name: label_ts3,
            data: data_ts
        })
    });
    $.getJSON(series_ts4, function(data_ts) {
        fracarea_chart.series[3].update({
            name: label_ts4,
            data: data_ts
        })
    });
    $.getJSON(series_ts5, function(data_ts) {
        fracarea_chart.series[4].update({
            name: label_ts5,
            data: data_ts
        })
    });

    // refresh button finished loading
    //$('#ts_drought_refresh').attr('src', './js/img/refresh.png');
    fracarea_chart.hideLoading();
    fracarea_chart.reflow();
    document.getElementById("county_stats_frac_refresh").style.display = 'none';

};

// ------- //





var date_in_saved_pap;
var bbox_in_saved_pap;
var image_link_pap_saved;
var date_in_saved_vci;
var bbox_in_saved_vci;
var image_link_vci_saved;
var date_in_saved_tci;
var bbox_in_saved_tci;
var image_link_tci_saved;
var date_in_saved_vhi;
var bbox_in_saved_vhi;
var image_link_vhi_saved;
var date_in_saved_nvai;
var bbox_in_saved_nvai;
var image_link_nvai_saved;
var date_in_saved_ntai;
var bbox_in_saved_ntai;
var image_link_ntai_saved;
var date_in_saved_netai;
var bbox_in_saved_netai;
var image_link_netai_saved;
var b;
var c;



    // --- // --- // --- // --- // --- // --- // --- // --- //
executePrecipitationAnomalyPercentageSLDQuery = function() {

    lattop = 53.7;
    latbottom = -11.6;
    longright = 141.1;
    longleft = 73.5;
    if ($('#startDate_map').length > 0) {
        // dostuff
        date_in = $('#startDate_map').MonthPicker('GetSelectedYear') + '-' + pad($(
            '#startDate_map').MonthPicker('GetSelectedMonth')) + '-' + '01';
    } else {
        date_in = '2015-01-01';
        console.log('date_in not undefined, set to default:');
    };
    if (c) {
        console.log('check before bbox_in_saved_pap: ' + JSON.stringify(bbox_in_saved_pap));
        var bbox_in = owl.deepCopy(c);
        console.log('can read bbox_in from b: ' + JSON.stringify(bbox_in));        
    } else {
        //console.log(e.message);
	var bbox_in = [longleft,latbottom,longright,lattop];
        console.log('set bbox_in to default:');
    };




// check if dataset is available based on metadata
if (typeof metadata_gpcp_to_date !== 'undefined') {
    // metadata_gpcp_to_date is defined
    // check if selected date is AFTER available date
    if (date_in > metadata_gpcp_to_date) {
        // your code here.
        var $message_metadata_gpcp = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The selected date is AFTER the available date! GPCP dataset availability: ' + metadata_gpcp_from_date + ' to ' + metadata_gpcp_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_gpcp.remove();
                    $('#map_di_pap_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_gpcp.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    // check if selected date is BEFORE available date
    if (date_in < metadata_gpcp_from_date) {
        // your code here.
        var $message_metadata_gpcp = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The selected date is BEFORE the available date! GPCP dataset availability: ' + metadata_gpcp_from_date + ' to ' + metadata_gpcp_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_gpcp.remove();
                    $('#map_di_pap_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_gpcp.dialog('open').parent().addClass("ui-state-error");
        return;
    };

    run_the_PAP_Query();

} else {
    // option include cancel or continue
    console.log('aaaw');
    function confirmation() { 
	console.log('checking...');
	var defer = $.Deferred();

	var $message_NO_metadata_gpcp = $('<div></div>')
	    .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No metadata available, click OK to continue or cancel to go back to safety')
	    .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
	    title: 'Warning',
	    buttons: {
	        Continue: function () {
	            defer.resolve("true");
	    	    $(this).dialog("close");
		    run_the_PAP_Query();
            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close");
                    $('#map_di_pap_refresh').attr('src', './js/img/refresh.png');
     	        },
	    },
	    close: function () {
		$(this).remove();
            }
	});

	$message_NO_metadata_gpcp.dialog('open').parent().addClass("ui-state-error");
	return defer.promise()

    };


    confirmation().then(function (answer) {
	console.log(answer);
	var ansbool = answer == "true";
	if (ansbool) {
	    console.log('continue as answer confirmation is ' + ansbool); //TRUE
	} else {
	    console.log('return as answer confirmation is ' + ansbool); //FALSE
	    return;
	}
    });    
};



function run_the_PAP_Query() {
    var date_in_ = JSON.stringify(date_in);
    var bbox_in_ = JSON.stringify(bbox_in);

    console.log('bbox_in: ' + bbox_in_);
    console.log('bbox_in_saved_pap: ' + bbox_in_saved_pap);
    console.log('date_in: ' + date_in_);
    console.log('date_in_saved_pap: ' + date_in_saved_pap);
    console.log('image_link_pap: ' + image_link_pap_saved);


    console.log(date_in_ === date_in_saved_pap);
    console.log(bbox_in_ === bbox_in_saved_pap);
    console.log(image_link_pap_saved);


    // If file is already computed, don't compute again, but load the saved URL
    if (date_in_ === date_in_saved_pap && bbox_in_ === bbox_in_saved_pap && image_link_pap_saved){ //

        // ▾▾▾▾▾
	console.log('function within IF statement PAP');

	var layers = map.getLayersByName("SLD test");
	for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
	map.removeLayer(layers[layerIndex]);
	};

	WMS_SLD = new OpenLayers.Layer.WMS("SLD test", image_link_pap_saved.slice(1,-1), {
	"layers": "map",
	"format": "image/png",
	"version": "1.1.1",
	"transparent": "TRUE",
	"SLD": "http://159.226.117.95:50080/drought/settings_mapserv/sld_raster_pap.xml"
	});
	WMS_SLD.setOpacity(0.7);
	map.addLayer(WMS_SLD);
	transform.deactivate(); //The remove the box with handles
	vectors.destroyFeatures();
	$('#map_di_pap_refresh').attr('src', './js/img/refresh.png');
	if ($('#legend_layer')) $('#legend_layer').remove();
	$('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link_pap_saved.slice(1,-1) +
	'&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95:50080%2Fdrought%2Fsettings_mapserv%2Fsld_raster_pap_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
	).appendTo('body'); // CGI Legend
	if ($('#date_layer')) $('#date_layer').remove();
	$('<div id="date_layer" class="date_layer"><b>Date: ' + date_in + '</b></div>').appendTo(
	'body'); // Date


        // ▴▴▴▴▴
    // If file is not yet computed execute PyWPS service    
    } else { // ▾▾▾▾▾      
	console.log('function within ELSE statement PAP');         

	// set the proxy
	// OpenLayers.ProxyHost = "/cgi-bin/proxy.cgi?url=";
	// init the client
	wps = new OpenLayers.WPS(urlWPS, {
	onSucceeded: onExecuted_PAP_DI
	});
	// define the process and append it to OpenLayers.WPS instance
	var PAP_SLD = new OpenLayers.WPS.Process({
	identifier: "WPS_PRECIP_DI_CAL",
	inputs: [
	    new OpenLayers.WPS.LiteralPut({
		identifier: "bbox",
		value: bbox_in
		    //asReference: true
	    }),
	    new OpenLayers.WPS.LiteralPut({
		identifier: "date",
		value: date_in
	    })
	],
	async: false,
	outputs: [
	    new OpenLayers.WPS.ComplexPut({
		identifier: "map",
		asReference: true
	    })
	]
	});

	date_in_saved_pap = JSON.stringify(date_in);
	bbox_in_saved_pap = JSON.stringify(bbox_in); 
        



	// defined earlier
	wps.addProcess(PAP_SLD);
	// run Execute
	wps.execute("WPS_PRECIP_DI_CAL");

    
    }; // ▴▴▴▴▴



};
};

// --- // --- // --- // --- // --- // --- // --- // --- //
var WMS_SLD
    /**

 * WPS events
 */
    // Everything went OK 
onExecuted_PAP_DI = function(process) {
    // Check if previous layer is computed, if so, remove it
    var layers = map.getLayersByName("SLD test");
    for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
        map.removeLayer(layers[layerIndex]);
    }
    image_out = process.outputs[0].getValue();
    image_out_slice = image_out.slice(43, 118);
    if (endsWith(image_out_slice, '%')) {
        console.log('image_slicing last character'),
            image_out_slice = image_out_slice.slice(0, -1)
    };
    if (endsWith(image_out_slice, '%2')) {
        console.log('image_slicing last character'),
            image_out_slice = image_out_slice.slice(0, -2)
    };
    console.log('image_out_slice: ' + image_out_slice);
    image_link = '../cgi-bin/mapserv?map='
    image_link += image_out_slice;
    image_link_pap_saved = JSON.stringify(image_link);
    //"http://localhost/cgi-bin/mapserv?map=/var/www/html/wpsoutputs/wps23332-tmpKCoWpM.map"
    console.log('image_out_slice: ' + image_link);
    WMS_SLD = new OpenLayers.Layer.WMS("SLD test", image_link, {
        "layers": "map",
        "format": "image/png",
        "version": "1.1.1",
        "transparent": "TRUE",
        "SLD": "http://159.226.117.95:50080/drought/settings_mapserv/sld_raster_pap.xml"
    });
    WMS_SLD.setOpacity(0.7);
    map.addLayer(WMS_SLD);
    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    $('#map_di_pap_refresh').attr('src', './js/img/refresh.png');
    if ($('#legend_layer')) $('#legend_layer').remove();
    $('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link +
        '&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95:50080%2Fdrought%2Fsettings_mapserv%2Fsld_raster_pap_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
    ).appendTo('body'); // CGI Legend
    if ($('#date_layer')) $('#date_layer').remove();
    $('<div id="date_layer" class="date_layer"><b>Date: ' + date_in + '</b></div>').appendTo(
        'body'); // Date






};

// --- // --- // --- // --- // ---

// --- // --- // --- // --- // --- // --- // --- // --- //
executeVCIQuery = function() {
    lattop = 53.7;
    latbottom = -11.6;
    longright = 141.1;
    longleft = 73.5;
    if ($('#startDate_map').length > 0) {
        // dostuff
        date_in = $('#startDate_map').MonthPicker('GetSelectedYear') + '-' + pad($(
            '#startDate_map').MonthPicker('GetSelectedMonth')) + '-' + '01';
    } else {
        date_in = '2015-01-01';
        console.log('date_in not undefined, set to default:');
    };
    if (c) {        
        var bbox_in = owl.deepCopy(c);
        console.log('can read bbox_in from b: ' + JSON.stringify(bbox_in));        
    } else {
        //console.log(e.message);
	var bbox_in = [longleft,latbottom,longright,lattop];
        console.log('set bbox_in to default:');
    };




// check if dataset is available based on metadata
if (typeof metadata_NDVI_MOD13C1005_to_date !== 'undefined') {
    // metadata_NDVI_MOD13C1005_to_date is defined
    // check if selected date is AFTER available date
    if (date_in > metadata_NDVI_MOD13C1005_to_date) {
        // your code here.
        var $message_metadata_NDVI_MOD13C1005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The selected date is AFTER the available date! NDVI dataset availability: ' + metadata_NDVI_MOD13C1005_from_date + ' to ' + metadata_NDVI_MOD13C1005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_NDVI_MOD13C1005.remove();
		    $('#map_di_vci_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_NDVI_MOD13C1005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    // check if selected date is BEFORE available date
    if (date_in < metadata_NDVI_MOD13C1005_from_date) {
        // your code here.
        var $message_metadata_NDVI_MOD13C1005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The selected date is BEFORE the available date! NDVI dataset availability: ' + metadata_NDVI_MOD13C1005_from_date + ' to ' + metadata_NDVI_MOD13C1005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_NDVI_MOD13C1005.remove();
		    $('#map_di_vci_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_NDVI_MOD13C1005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    run_the_VCI_Query();

} else {
    // option include cancel or continue
    console.log('aaaw');
    function confirmation() { 
	console.log('checking...');
	var defer = $.Deferred();

	var $message_NO_metadata_NDVI_MOD13C1005 = $('<div></div>')
	    .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No metadata available, click OK to continue or cancel to go back to safety')
	    .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
	    title: 'Warning',
	    buttons: {
	        Continue: function () {
	            defer.resolve("true");
	    	    $(this).dialog("close");
		    run_the_VCI_Query();
            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close");
                    $('#map_di_vci_refresh').attr('src', './js/img/refresh.png');
     	        },
	    },
	    close: function () {
		$(this).remove();
            }
	});

	$message_NO_metadata_NDVI_MOD13C1005.dialog('open').parent().addClass("ui-state-error");
	return defer.promise()

    };


    confirmation().then(function (answer) {
	console.log(answer);
	var ansbool = answer == "true";
	if (ansbool) {
	    console.log('continue as answer confirmation is ' + ansbool); //TRUE
	} else {
	    console.log('return as answer confirmation is ' + ansbool); //FALSE
	    return;
	}
    });    
};




function run_the_VCI_Query() {
    var date_in_ = JSON.stringify(date_in);
    var bbox_in_ = JSON.stringify(bbox_in);

    console.log('bbox_in: ' + bbox_in_);
    console.log('bbox_in_saved_vci: ' + bbox_in_saved_vci);
    console.log('date_in: ' + date_in_);
    console.log('date_in_saved_vci: ' + date_in_saved_vci);
    console.log('image_link_vci: ' + image_link_vci_saved);


    console.log(date_in_ === date_in_saved_vci);
    console.log(bbox_in_ === bbox_in_saved_vci);
    console.log(image_link_vci_saved);


    // If file is already computed, don't compute again, but load the saved URL
    if (date_in_ === date_in_saved_vci && bbox_in_ === bbox_in_saved_vci && image_link_vci_saved){ //

        // ▾▾▾▾▾
	console.log('function within IF statement VCI');

	var layers = map.getLayersByName("SLD test");
	for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
	map.removeLayer(layers[layerIndex]);
	};

	WMS_SLD = new OpenLayers.Layer.WMS("SLD test", image_link_vci_saved.slice(1,-1), {
	"layers": "map",
	"format": "image/png",
	"version": "1.1.1",
	"transparent": "TRUE",
	"SLD": "http://159.226.117.95:50080/drought/settings_mapserv/sld_raster_vci.xml"
	});
	WMS_SLD.setOpacity(0.7);
	map.addLayer(WMS_SLD);
	transform.deactivate(); //The remove the box with handles
	vectors.destroyFeatures();
	$('#map_di_vci_refresh').attr('src', './js/img/refresh.png');
	if ($('#legend_layer')) $('#legend_layer').remove();
	$('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link_vci_saved.slice(1,-1) +
	'&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95:50080%2Fdrought%2Fsettings_mapserv%2Fsld_raster_vci_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
	).appendTo('body'); // CGI Legend
	if ($('#date_layer')) $('#date_layer').remove();
	$('<div id="date_layer" class="date_layer"><b>Date: ' + date_in + '</b></div>').appendTo(
	'body'); // Date


        // ▴▴▴▴▴
    // If file is not yet computed execute PyWPS service    
    } else { // ▾▾▾▾▾      
	console.log('function within ELSE statement VCI');         



	// set the proxy
	// OpenLayers.ProxyHost = "/cgi-bin/proxy.cgi?url=";
	// init the client
	wps = new OpenLayers.WPS(urlWPS, {
	onSucceeded: onExecuted_VCI_DI
	});
	// define the process and append it to OpenLayers.WPS instance
	var VCI_SLD = new OpenLayers.WPS.Process({
	identifier: "WPS_VCI_DI_CAL",
	inputs: [
	    new OpenLayers.WPS.LiteralPut({
		identifier: "bbox",
		value: bbox_in
		    //asReference: true
	    }),
	    new OpenLayers.WPS.LiteralPut({
		identifier: "date",
		value: date_in
	    })
	],
	async: false,
	outputs: [
	    new OpenLayers.WPS.ComplexPut({
		identifier: "map",
		asReference: true
	    })
	]
	});

	date_in_saved_vci = JSON.stringify(date_in); //JSON.parse(JSON.stringify(date_in)) ;
	bbox_in_saved_vci = JSON.stringify(bbox_in); //bbox_in.slice(0);//JSON.parse(JSON.stringify(bbox_in,4,null));  /

	// defined earlier
	wps.addProcess(VCI_SLD);
	// run Execute
	wps.execute("WPS_VCI_DI_CAL");

    }; // ▴▴▴▴▴
};
};
// --- // --- // --- // --- // --- // --- // --- // --- //
var WMS_SLD
    /**


 * WPS events
 */
    // Everything went OK 
onExecuted_VCI_DI = function(process) {
    // Check if previous layer is computed, if so, remove it
    console.log('we are already here!');
    var layers = map.getLayersByName("SLD test");
    for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
        map.removeLayer(layers[layerIndex]);
    }
    image_out = process.outputs[0].getValue();
    image_out_slice = image_out.slice(43, 118);
    if (endsWith(image_out_slice, '%')) {
        console.log('image_slicing last character'),
            image_out_slice = image_out_slice.slice(0, -1)
    };
    if (endsWith(image_out_slice, '%2')) {
        console.log('image_slicing last character'),
            image_out_slice = image_out_slice.slice(0, -2)
    };
    console.log('image_out_slice: ' + image_out_slice);
    image_link = '../cgi-bin/mapserv?map='
    image_link += image_out_slice;
    image_link_vci_saved = JSON.stringify(image_link);
    //"http://localhost/cgi-bin/mapserv?map=/var/www/html/wpsoutputs/wps23332-tmpKCoWpM.map"
    console.log('image_out_slice: ' + image_link);
    WMS_SLD = new OpenLayers.Layer.WMS("SLD test", image_link, {
        "layers": "map",
        "format": "image/png",
        "version": "1.1.1",
        "transparent": "TRUE",
        "SLD": "http://159.226.117.95:50080/drought/settings_mapserv/sld_raster_vci.xml"
    });
    WMS_SLD.setOpacity(0.7);
    map.addLayer(WMS_SLD);
    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    $('#map_di_vci_refresh').attr('src', './js/img/refresh.png');
    if ($('#legend_layer')) $('#legend_layer').remove();
    $('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link +
        '&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95:50080%2Fdrought%2Fsettings_mapserv%2Fsld_raster_vci_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
    ).appendTo('body'); // CGI Legend
    if ($('#date_layer')) $('#date_layer').remove();
    $('<div id="date_layer" class="date_layer"><b>Date: ' + date_in + '</b></div>').appendTo(
        'body'); // Date
    


};

// --- // --- // --- // --- // ---
// --- // --- // --- // --- // --- // --- // --- // --- //
executeTCIQuery = function() {
    lattop = 53.7;
    latbottom = -11.6;
    longright = 141.1;
    longleft = 73.5;
    if ($('#startDate_map').length > 0) {
        // dostuff
        date_in = $('#startDate_map').MonthPicker('GetSelectedYear') + '-' + pad($(
            '#startDate_map').MonthPicker('GetSelectedMonth')) + '-' + '01';
    } else {
        date_in = '2015-01-01';
        console.log('date_in not undefined, set to default:');
    };
    if (c) {        
        var bbox_in = owl.deepCopy(c);
        console.log('can read bbox_in from b: ' + JSON.stringify(bbox_in));        
    } else {
        //console.log(e.message);
	var bbox_in = [longleft,latbottom,longright,lattop];
        console.log('set bbox_in to default:');
    };





// check if dataset is available based on metadata
if (typeof metadata_LST_MOD11C2005_to_date !== 'undefined') {
    // metadata_LST_MOD11C2005_to_date is defined
    // check if selected date is AFTER available date
    if (date_in > metadata_LST_MOD11C2005_to_date) {
        // your code here.
        var $message_metadata_LST_MOD11C2005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The selected date is AFTER the available date! Dataset availability: ' + metadata_LST_MOD11C2005_from_date + ' to ' + metadata_LST_MOD11C2005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_LST_MOD11C2005.remove();
                    $('#map_di_tci_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_LST_MOD11C2005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    // check if selected date is BEFORE available date
    if (date_in < metadata_LST_MOD11C2005_from_date) {
        // your code here.
        var $message_metadata_LST_MOD11C2005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The selected date is BEFORE the available date! Dataset availability: ' + metadata_LST_MOD11C2005_from_date + ' to ' + metadata_LST_MOD11C2005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_LST_MOD11C2005.remove();
                    $('#map_di_tci_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_LST_MOD11C2005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    run_the_TCI_Query();

} else {
    // option include cancel or continue
    console.log('aaaw');
    function confirmation() { 
	console.log('checking...');
	var defer = $.Deferred();

	var $message_NO_metadata_LST_MOD11C2005 = $('<div></div>')
	    .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No metadata available, click OK to continue or cancel to go back to safety')
	    .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
	    title: 'Warning',
	    buttons: {
	        Continue: function () {
	            defer.resolve("true");
	    	    $(this).dialog("close");
		    run_the_TCI_Query();
            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close");
                    $('#map_di_tci_refresh').attr('src', './js/img/refresh.png');
     	        },
	    },
	    close: function () {
		$(this).remove();
            }
	});

	$message_NO_metadata_LST_MOD11C2005.dialog('open').parent().addClass("ui-state-error");
	return defer.promise()

    };


    confirmation().then(function (answer) {
	console.log(answer);
	var ansbool = answer == "true";
	if (ansbool) {
	    console.log('continue as answer confirmation is ' + ansbool); //TRUE
	} else {
	    console.log('return as answer confirmation is ' + ansbool); //FALSE
	    return;
	}
    });    
};



function run_the_TCI_Query() {

    var date_in_ = JSON.stringify(date_in);
    var bbox_in_ = JSON.stringify(bbox_in);

    console.log('bbox_in: ' + bbox_in_);
    console.log('bbox_in_saved_tci: ' + bbox_in_saved_tci);
    console.log('date_in: ' + date_in_);
    console.log('date_in_saved_tci: ' + date_in_saved_tci);
    console.log('image_link_tci: ' + image_link_tci_saved);


    console.log(date_in_ === date_in_saved_tci);
    console.log(bbox_in_ === bbox_in_saved_tci);
    console.log(image_link_tci_saved);


    // If file is already computed, don't compute again, but load the saved URL
    if (date_in_ === date_in_saved_tci && bbox_in_ === bbox_in_saved_tci && image_link_tci_saved){ //

        // ▾▾▾▾▾
	console.log('function within IF statement TCI');

	var layers = map.getLayersByName("SLD test");
	for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
	map.removeLayer(layers[layerIndex]);
	};

	WMS_SLD = new OpenLayers.Layer.WMS("SLD test", image_link_tci_saved.slice(1,-1), {
	"layers": "map",
	"format": "image/png",
	"version": "1.1.1",
	"transparent": "TRUE",
	"SLD": "http://159.226.117.95:50080/drought/settings_mapserv/sld_raster_tci.xml"
	});
	WMS_SLD.setOpacity(0.7);
	map.addLayer(WMS_SLD);
	transform.deactivate(); //The remove the box with handles
	vectors.destroyFeatures();
	$('#map_di_tci_refresh').attr('src', './js/img/refresh.png');
	if ($('#legend_layer')) $('#legend_layer').remove();
	$('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link_tci_saved.slice(1,-1) +
	'&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95:50080%2Fdrought%2Fsettings_mapserv%2Fsld_raster_tci_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
	).appendTo('body'); // CGI Legend
	if ($('#date_layer')) $('#date_layer').remove();
	$('<div id="date_layer" class="date_layer"><b>Date: ' + date_in + '</b></div>').appendTo(
	'body'); // Date




        // ▴▴▴▴▴
    // If file is not yet computed execute PyWPS service    
    } else { // ▾▾▾▾▾      
	console.log('function within ELSE statement TCI');      
	// set the proxy
	// OpenLayers.ProxyHost = "/cgi-bin/proxy.cgi?url=";
	// init the client
	wps = new OpenLayers.WPS(urlWPS, {
	onSucceeded: onExecuted_TCI_DI
	});
	// define the process and append it to OpenLayers.WPS instance
	var TCI_SLD = new OpenLayers.WPS.Process({
	identifier: "WPS_TCI_DI_CAL",
	inputs: [
	    new OpenLayers.WPS.LiteralPut({
		identifier: "bbox",
		value: bbox_in
		    //asReference: true
	    }),
	    new OpenLayers.WPS.LiteralPut({
		identifier: "date",
		value: date_in
	    })
	],
	async: false,
	outputs: [
	    new OpenLayers.WPS.ComplexPut({
		identifier: "map",
		asReference: true
	    })
	]
	});

	date_in_saved_tci = JSON.stringify(date_in);        
	bbox_in_saved_tci = JSON.stringify(bbox_in); 

	// defined earlier
	wps.addProcess(TCI_SLD);
	// run Execute
	wps.execute("WPS_TCI_DI_CAL");	
    }; // ▴▴▴▴▴
};
};
// --- // --- // --- // --- // --- // --- // --- // --- //
var WMS_SLD
    /**

 * WPS events
 */
    // Everything went OK 
onExecuted_TCI_DI = function(process) {
    // Check if previous layer is computed, if so, remove it
    console.log('we are already here!');
    var layers = map.getLayersByName("SLD test");
    for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
        map.removeLayer(layers[layerIndex]);
    }
    image_out = process.outputs[0].getValue();
    image_out_slice = image_out.slice(43, 118);
    if (endsWith(image_out_slice, '%')) {
        console.log('image_slicing last character'),
            image_out_slice = image_out_slice.slice(0, -1)
    };
    if (endsWith(image_out_slice, '%2')) {
        console.log('image_slicing last character'),
            image_out_slice = image_out_slice.slice(0, -2)
    };
    console.log('image_out_slice: ' + image_out_slice);
    image_link = '../cgi-bin/mapserv?map='
    image_link += image_out_slice;
    image_link_tci_saved = JSON.stringify(image_link);
    //"http://localhost/cgi-bin/mapserv?map=/var/www/html/wpsoutputs/wps23332-tmpKCoWpM.map"
    console.log('image_out_slice: ' + image_link);
    WMS_SLD = new OpenLayers.Layer.WMS("SLD test", image_link, {
        "layers": "map",
        "format": "image/png",
        "version": "1.1.1",
        "transparent": "TRUE",
        "SLD": "http://159.226.117.95:50080/drought/settings_mapserv/sld_raster_tci.xml"
    });
    WMS_SLD.setOpacity(0.7);
    map.addLayer(WMS_SLD);

    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    $('#map_di_tci_refresh').attr('src', './js/img/refresh.png');
    if ($('#legend_layer')) $('#legend_layer').remove();
    $('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link +
        '&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95:50080%2Fdrought%2Fsettings_mapserv%2Fsld_raster_tci_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
    ).appendTo('body'); // CGI Legend
    if ($('#date_layer')) $('#date_layer').remove();
    $('<div id="date_layer" class="date_layer"><b>Date: ' + date_in + '</b></div>').appendTo(
        'body'); // Date
};

// --- // --- // --- // --- // ---

// VHI // --- // --- // --- // --- // --- // --- // --- //
executeVHIQuery = function() {
    lattop = 53.7;
    latbottom = -11.6;
    longright = 141.1;
    longleft = 73.5;
    if ($('#startDate_map').length > 0) {
        // dostuff
        date_in = $('#startDate_map').MonthPicker('GetSelectedYear') + '-' + pad($(
            '#startDate_map').MonthPicker('GetSelectedMonth')) + '-' + '01';
    } else {
        date_in = '2015-01-01';
        console.log('date_in not undefined, set to default:');
    };
    if (c) {        
        var bbox_in = owl.deepCopy(c);
        console.log('can read bbox_in from b: ' + JSON.stringify(bbox_in));        
    } else {
        //console.log(e.message);
	var bbox_in = [longleft,latbottom,longright,lattop];
        console.log('set bbox_in to default:');
    };



// check if dataset is available based on metadata
if (typeof metadata_LST_MOD11C2005_to_date !== 'undefined') {
    // metadata_LST_MOD11C2005_to_date is defined
    // check if selected date is AFTER available date
    if (date_in > metadata_LST_MOD11C2005_to_date) {
        // your code here.
        var $message_metadata_LST_MOD11C2005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The index is based on two datasets, the selected date of LST is AFTER the available date! Dataset availability: ' + metadata_LST_MOD11C2005_from_date + ' to ' + metadata_LST_MOD11C2005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_LST_MOD11C2005.remove();
                    $('#map_di_vhi_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_LST_MOD11C2005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    // check if selected date is BEFORE available date
    if (date_in < metadata_LST_MOD11C2005_from_date) {
        // your code here.
        var $message_metadata_LST_MOD11C2005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The index is based on two datasets, the selected date of LST is BEFORE the available date! Dataset availability: ' + metadata_LST_MOD11C2005_from_date + ' to ' + metadata_LST_MOD11C2005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_LST_MOD11C2005.remove();
                    $('#map_di_vhi_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_LST_MOD11C2005.dialog('open').parent().addClass("ui-state-error");
        return;
    };

    //run_the_VHI_Query();

} else {
    // option include cancel or continue
    console.log('aaaw');
    function confirmation() { 
	console.log('checking...');
	var defer = $.Deferred();

	var $message_NO_metadata_LST_MOD11C2005 = $('<div></div>')
	    .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No metadata available, click OK to continue or cancel to go back to safety')
	    .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
	    title: 'Warning',
	    buttons: {
	        Continue: function () {
	            defer.resolve("true");
	    	    $(this).dialog("close");
		    //run_the_VHI_Query();
            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close");
                    $('#map_di_vhi_refresh').attr('src', './js/img/refresh.png');
     	        },
	    },
	    close: function () {
		$(this).remove();
            }
	});

	$message_NO_metadata_LST_MOD11C2005.dialog('open').parent().addClass("ui-state-error");
	return defer.promise()

    };


    confirmation().then(function (answer) {
	console.log(answer);
	var ansbool = answer == "true";
	if (ansbool) {
	    console.log('continue as answer confirmation is ' + ansbool); //TRUE
	} else {
	    console.log('return as answer confirmation is ' + ansbool); //FALSE
	    return;
	}
    });    
};





// check if dataset is available based on metadata
if (typeof metadata_NDVI_MOD13C1005_to_date !== 'undefined') {
    // metadata_NDVI_MOD13C1005_to_date is defined
    // check if selected date is AFTER available date
    if (date_in > metadata_NDVI_MOD13C1005_to_date) {
        // your code here.
        var $message_metadata_NDVI_MOD13C1005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The index is based on two datasets, the start date of NDVI is AFTER the available date! Dataset availability: ' + metadata_NDVI_MOD13C1005_from_date + ' to ' + metadata_NDVI_MOD13C1005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_NDVI_MOD13C1005.remove();
                    $('#map_di_vhi_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_NDVI_MOD13C1005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    // check if selected date is BEFORE available date
    if (date_in < metadata_NDVI_MOD13C1005_from_date) {
        // your code here.
        var $message_metadata_NDVI_MOD13C1005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The index is based on two datasets, the start date of NDVI is BEFORE the available date! Dataset availability: ' + metadata_NDVI_MOD13C1005_from_date + ' to ' + metadata_NDVI_MOD13C1005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_NDVI_MOD13C1005.remove();
                    $('#map_di_vhi_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_NDVI_MOD13C1005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    run_the_VHI_Query();

} else {
    // option include cancel or continue
    console.log('aaaw');
    function confirmation() { 
	console.log('checking...');
	var defer = $.Deferred();

	var $message_NO_metadata_NDVI_MOD13C1005 = $('<div></div>')
	    .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No metadata available, click OK to continue or cancel to go back to safety')
	    .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
	    title: 'Warning',
	    buttons: {
	        Continue: function () {
	            defer.resolve("true");
	    	    $(this).dialog("close");
		    run_the_VHI_Query();
            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close");
                    $('#map_di_vhi_refresh').attr('src', './js/img/refresh.png');
     	        },
	    },
	    close: function () {
		$(this).remove();
            }
	});

	$message_NO_metadata_NDVI_MOD13C1005.dialog('open').parent().addClass("ui-state-error");
	return defer.promise()

    };


    confirmation().then(function (answer) {
	console.log(answer);
	var ansbool = answer == "true";
	if (ansbool) {
	    console.log('continue as answer confirmation is ' + ansbool); //TRUE
	} else {
	    console.log('return as answer confirmation is ' + ansbool); //FALSE
	    return;
	}
    });    
};





function run_the_VHI_Query(){
    var date_in_ = JSON.stringify(date_in);
    var bbox_in_ = JSON.stringify(bbox_in);

    console.log('bbox_in: ' + bbox_in_);
    console.log('bbox_in_saved_vhi: ' + bbox_in_saved_vhi);
    console.log('date_in: ' + date_in_);
    console.log('date_in_saved_vhi: ' + date_in_saved_vhi);
    console.log('image_link_vhi: ' + image_link_vhi_saved);


    console.log(date_in_ === date_in_saved_vhi);
    console.log(bbox_in_ === bbox_in_saved_vhi);
    console.log(image_link_vhi_saved);


    // If file is already computed, don't compute again, but load the saved URL
    if (date_in_ === date_in_saved_vhi && bbox_in_ === bbox_in_saved_vhi && image_link_vhi_saved){ //
        // ▾▾▾▾▾
	console.log('function within IF statement VHI');

	var layers = map.getLayersByName("SLD test");
	for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
	map.removeLayer(layers[layerIndex]);
	};

	WMS_SLD = new OpenLayers.Layer.WMS("SLD test", image_link_vhi_saved.slice(1,-1), {
	"layers": "map",
	"format": "image/png",
	"version": "1.1.1",
	"transparent": "TRUE",
	"SLD": "http://159.226.117.95:50080/drought/settings_mapserv/sld_raster_vhi.xml"
	});
	WMS_SLD.setOpacity(0.7);
	map.addLayer(WMS_SLD);
	transform.deactivate(); //The remove the box with handles
	vectors.destroyFeatures();
	$('#map_di_vhi_refresh').attr('src', './js/img/refresh.png');
	if ($('#legend_layer')) $('#legend_layer').remove();
	$('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link_vhi_saved.slice(1,-1) +
	'&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95:50080%2Fdrought%2Fsettings_mapserv%2Fsld_raster_vhi_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
	).appendTo('body'); // CGI Legend
	if ($('#date_layer')) $('#date_layer').remove();
	$('<div id="date_layer" class="date_layer"><b>Date: ' + date_in + '</b></div>').appendTo(
	'body'); // Date



        // ▴▴▴▴▴
    // If file is not yet computed execute PyWPS service    
    } else { // ▾▾▾▾▾      
	console.log('function within ELSE statement VHI');   
    // set the proxy
    // OpenLayers.ProxyHost = "/cgi-bin/proxy.cgi?url=";
    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecuted_VHI_DI
    });
    // define the process and append it to OpenLayers.WPS instance
    var VHI_SLD = new OpenLayers.WPS.Process({
        identifier: "WPS_VHI_DI_CAL",
        inputs: [
            new OpenLayers.WPS.LiteralPut({
                identifier: "bbox",
                value: bbox_in
                    //asReference: true
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "date",
                value: date_in
            })
        ],
        async: false,
        outputs: [
            new OpenLayers.WPS.ComplexPut({
                identifier: "map",
                asReference: true
            })
        ]
    });

	date_in_saved_vhi = JSON.stringify(date_in);        
	bbox_in_saved_vhi = JSON.stringify(bbox_in); 

    // defined earlier
    wps.addProcess(VHI_SLD);
    // run Execute
    wps.execute("WPS_VHI_DI_CAL");
    }; // ▴▴▴▴▴
};
};
// --- // --- // --- // --- // --- // --- // --- // --- //
var WMS_SLD
    /**

 * WPS events
 */
    // Everything went OK 
onExecuted_VHI_DI = function(process) {
    // Check if previous layer is computed, if so, remove it
    console.log('we are already here!');
    var layers = map.getLayersByName("SLD test");
    for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
        map.removeLayer(layers[layerIndex]);
    }
    image_out = process.outputs[0].getValue();
    image_out_slice = image_out.slice(43, 118);
    if (endsWith(image_out_slice, '%')) {
        console.log('image_slicing last character'),
            image_out_slice = image_out_slice.slice(0, -1)
    };
    if (endsWith(image_out_slice, '%2')) {
        console.log('image_slicing last character'),
            image_out_slice = image_out_slice.slice(0, -2)
    };
    console.log('image_out_slice: ' + image_out_slice);
    image_link = '../cgi-bin/mapserv?map='
    image_link += image_out_slice;
    image_link_vhi_saved = JSON.stringify(image_link);
    //"http://localhost/cgi-bin/mapserv?map=/var/www/html/wpsoutputs/wps23332-tmpKCoWpM.map"
    console.log('image_out_slice: ' + image_link);
    WMS_SLD = new OpenLayers.Layer.WMS("SLD test", image_link, {
        "layers": "map",
        "format": "image/png",
        "version": "1.1.1",
        "transparent": "TRUE",
        "SLD": "http://159.226.117.95:50080/drought/settings_mapserv/sld_raster_vhi.xml"
    });
    WMS_SLD.setOpacity(0.7);
    map.addLayer(WMS_SLD);

    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    $('#map_di_vhi_refresh').attr('src', './js/img/refresh.png');
    if ($('#legend_layer')) $('#legend_layer').remove();
    $('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link +
        '&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95:50080%2Fdrought%2Fsettings_mapserv%2Fsld_raster_vhi_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
    ).appendTo('body'); // CGI Legend
    if ($('#date_layer')) $('#date_layer').remove();
    $('<div id="date_layer" class="date_layer"><b>Date: ' + date_in + '</b></div>').appendTo(
        'body'); // Date
};

// --- // --- // --- // --- // ---


// NVAI // --- // --- // --- // --- // --- // --- // --- //
executeNVAIQuery = function() {
    lattop = 53.7;
    latbottom = -11.6;
    longright = 141.1;
    longleft = 73.5;
    if ($('#startDate_map').length > 0) {
        // dostuff
        date_in = $('#startDate_map').MonthPicker('GetSelectedYear') + '-' + pad($(
            '#startDate_map').MonthPicker('GetSelectedMonth')) + '-' + '01';
    } else {
        date_in = '2015-01-01';
        console.log('date_in not undefined, set to default:');
    };
    if (c) {        
        var bbox_in = owl.deepCopy(c);
        console.log('can read bbox_in from b: ' + JSON.stringify(bbox_in));        
    } else {
        //console.log(e.message);
	var bbox_in = [longleft,latbottom,longright,lattop];
        console.log('set bbox_in to default:');
    };




// check if dataset is available based on metadata
if (typeof metadata_NDVI_MOD13C1005_to_date !== 'undefined') {
    // metadata_NDVI_MOD13C1005_to_date is defined
    // check if selected date is AFTER available date
    if (date_in > metadata_NDVI_MOD13C1005_to_date) {
        // your code here.
        var $message_metadata_NDVI_MOD13C1005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The selected date is AFTER the available date! NDVI dataset availability: ' + metadata_NDVI_MOD13C1005_from_date + ' to ' + metadata_NDVI_MOD13C1005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_NDVI_MOD13C1005.remove();
                    $('#map_di_nvai_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_NDVI_MOD13C1005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    // check if selected date is AFTER available date
    if (date_in < metadata_NDVI_MOD13C1005_from_date) {
        // your code here.
        var $message_metadata_NDVI_MOD13C1005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The selected date is BEFORE the available date! NDVI dataset availability: ' + metadata_NDVI_MOD13C1005_from_date + ' to ' + metadata_NDVI_MOD13C1005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_NDVI_MOD13C1005.remove();
                    $('#map_di_nvai_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_NDVI_MOD13C1005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    run_the_NVAI_Query();

} else {
    // option include cancel or continue
    console.log('aaaw');
    function confirmation() { 
	console.log('checking...');
	var defer = $.Deferred();

	var $message_NO_metadata_NDVI_MOD13C1005 = $('<div></div>')
	    .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No metadata available, click OK to continue or cancel to go back to safety')
	    .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
	    title: 'Warning',
	    buttons: {
	        Continue: function () {
	            defer.resolve("true");
	    	    $(this).dialog("close");
		    run_the_NVAI_Query();
            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close");
                    $('#map_di_nvai_refresh').attr('src', './js/img/refresh.png');
     	        },
	    },
	    close: function () {
		$(this).remove();
            }
	});

	$message_NO_metadata_NDVI_MOD13C1005.dialog('open').parent().addClass("ui-state-error");
	return defer.promise()

    };


    confirmation().then(function (answer) {
	console.log(answer);
	var ansbool = answer == "true";
	if (ansbool) {
	    console.log('continue as answer confirmation is ' + ansbool); //TRUE
	} else {
	    console.log('return as answer confirmation is ' + ansbool); //FALSE
	    return;
	}
    });    
};




function run_the_NVAI_Query() {
    var date_in_ = JSON.stringify(date_in);
    var bbox_in_ = JSON.stringify(bbox_in);

    console.log('bbox_in: ' + bbox_in_);
    console.log('bbox_in_saved_nvai: ' + bbox_in_saved_nvai);
    console.log('date_in: ' + date_in_);
    console.log('date_in_saved_nvai: ' + date_in_saved_nvai);
    console.log('image_link_nvai: ' + image_link_nvai_saved);


    console.log(date_in_ === date_in_saved_nvai);
    console.log(bbox_in_ === bbox_in_saved_nvai);
    console.log(image_link_nvai_saved);


    // If file is already computed, don't compute again, but load the saved URL
    if (date_in_ === date_in_saved_nvai && bbox_in_ === bbox_in_saved_nvai && image_link_nvai_saved){ //
        // ▾▾▾▾▾
	console.log('function within IF statement NVAI');

	var layers = map.getLayersByName("SLD test");
	for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
	map.removeLayer(layers[layerIndex]);
	};

	WMS_SLD = new OpenLayers.Layer.WMS("SLD test", image_link_nvai_saved.slice(1,-1), {
	"layers": "map",
	"format": "image/png",
	"version": "1.1.1",
	"transparent": "TRUE",
	"SLD": "http://159.226.117.95:50080/drought/settings_mapserv/sld_raster_nvai.xml"
	});
	WMS_SLD.setOpacity(0.7);
	map.addLayer(WMS_SLD);
	transform.deactivate(); //The remove the box with handles
	vectors.destroyFeatures();
	$('#map_di_nvai_refresh').attr('src', './js/img/refresh.png');
	if ($('#legend_layer')) $('#legend_layer').remove();
	$('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link_nvai_saved.slice(1,-1) +
	'&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95:50080%2Fdrought%2Fsettings_mapserv%2Fsld_raster_nvai_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
	).appendTo('body'); // CGI Legend
	if ($('#date_layer')) $('#date_layer').remove();
	$('<div id="date_layer" class="date_layer"><b>Date: ' + date_in + '</b></div>').appendTo(
	'body'); // Date




        // ▴▴▴▴▴
    // If file is not yet computed execute PyWPS service    
    } else { // ▾▾▾▾▾      
	console.log('function within ELSE statement NVAI');   
    // set the proxy
    // OpenLayers.ProxyHost = "/cgi-bin/proxy.cgi?url=";
    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecuted_NVAI_DI
    });
    // define the process and append it to OpenLayers.WPS instance
    var NVAI_SLD = new OpenLayers.WPS.Process({
        identifier: "WPS_NVAI_DI_CAL",
        inputs: [
            new OpenLayers.WPS.LiteralPut({
                identifier: "bbox",
                value: bbox_in
                    //asReference: true
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "date",
                value: date_in
            })
        ],
        async: false,
        outputs: [
            new OpenLayers.WPS.ComplexPut({
                identifier: "map",
                asReference: true
            })
        ]
    });

	date_in_saved_nvai = JSON.stringify(date_in);        
	bbox_in_saved_nvai = JSON.stringify(bbox_in); 

    // defined earlier
    wps.addProcess(NVAI_SLD);
    // run Execute
    wps.execute("WPS_NVAI_DI_CAL");
    }; // ▴▴▴▴▴
};
};
// --- // --- // --- // --- // --- // --- // --- // --- //
var WMS_SLD
    /**

 * WPS events
 */
    // Everything went OK 
onExecuted_NVAI_DI = function(process) {
    // Check if previous layer is computed, if so, remove it
    console.log('we are already here!');
    var layers = map.getLayersByName("SLD test");
    for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
        map.removeLayer(layers[layerIndex]);
    }
    image_out = process.outputs[0].getValue();
    image_out_slice = image_out.slice(43, 118);
    if (endsWith(image_out_slice, '%')) {
        console.log('image_slicing last character'),
            image_out_slice = image_out_slice.slice(0, -1)
    };
    if (endsWith(image_out_slice, '%2')) {
        console.log('image_slicing last character'),
            image_out_slice = image_out_slice.slice(0, -2)
    };
    console.log('image_out_slice: ' + image_out_slice);
    image_link = '../cgi-bin/mapserv?map='
    image_link += image_out_slice;
    image_link_nvai_saved = JSON.stringify(image_link);
    //"http://localhost/cgi-bin/mapserv?map=/var/www/html/wpsoutputs/wps23332-tmpKCoWpM.map"
    console.log('image_out_slice: ' + image_link);
    WMS_SLD = new OpenLayers.Layer.WMS("SLD test", image_link, {
        "layers": "map",
        "format": "image/png",
        "version": "1.1.1",
        "transparent": "TRUE",
        "SLD": "http://159.226.117.95:50080/drought/settings_mapserv/sld_raster_nvai.xml"
    });
    WMS_SLD.setOpacity(0.7);
    map.addLayer(WMS_SLD);

    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    $('#map_di_nvai_refresh').attr('src', './js/img/refresh.png');
    if ($('#legend_layer')) $('#legend_layer').remove();
    $('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link +
        '&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95:50080%2Fdrought%2Fsettings_mapserv%2Fsld_raster_nvai_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
    ).appendTo('body'); // CGI Legend
    if ($('#date_layer')) $('#date_layer').remove();
    $('<div id="date_layer" class="date_layer"><b>Date: ' + date_in + '</b></div>').appendTo(
        'body'); // Date
};

// --- // --- // --- // --- // ---

// NTAI // --- // --- // --- // --- // --- // --- // --- //
executeNTAIQuery = function() {
    lattop = 53.7;
    latbottom = -11.6;
    longright = 141.1;
    longleft = 73.5;
    if ($('#startDate_map').length > 0) {
        // dostuff
        date_in = $('#startDate_map').MonthPicker('GetSelectedYear') + '-' + pad($(
            '#startDate_map').MonthPicker('GetSelectedMonth')) + '-' + '01';
    } else {
        date_in = '2015-01-01';
        console.log('date_in not undefined, set to default:');
    };
    if (c) {        
        var bbox_in = owl.deepCopy(c);
        console.log('can read bbox_in from b: ' + JSON.stringify(bbox_in));        
    } else {
        //console.log(e.message);
	var bbox_in = [longleft,latbottom,longright,lattop];
        console.log('set bbox_in to default:');
    };




// check if dataset is available based on metadata
if (typeof metadata_LST_MOD11C2005_to_date !== 'undefined') {
    // metadata_LST_MOD11C2005_to_date is defined
    // check if selected date is AFTER available date
    if (date_in > metadata_LST_MOD11C2005_to_date) {
        // your code here.
        var $message_metadata_LST_MOD11C2005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The selected date is AFTER the available date! LST dataset availability: ' + metadata_LST_MOD11C2005_from_date + ' to ' + metadata_LST_MOD11C2005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_LST_MOD11C2005.remove();
                    $('#map_di_ntai_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_LST_MOD11C2005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    // check if selected date is BEFORE available date
    if (date_in < metadata_LST_MOD11C2005_from_date) {
        // your code here.
        var $message_metadata_LST_MOD11C2005 = $('<div></div>')
            .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> The selected date is BEFORE the available date! LST dataset availability: ' + metadata_LST_MOD11C2005_from_date + ' to ' + metadata_LST_MOD11C2005_to_date)
            .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
            title: 'Warning',
            buttons: {
                OK: function () {
                    $(this).dialog("close");
                    $message_metadata_LST_MOD11C2005.remove();
                    $('#map_di_ntai_refresh').attr('src', './js/img/refresh.png');
                }
            }
        });
        $message_metadata_LST_MOD11C2005.dialog('open').parent().addClass("ui-state-error");
        return;
    };
    run_the_NTAI_Query();

} else {
    // option include cancel or continue
    console.log('aaaw');
    function confirmation() { 
	console.log('checking...');
	var defer = $.Deferred();

	var $message_NO_metadata_LST_MOD11C2005 = $('<div></div>')
	    .html('<span class="ui-icon ui-icon-alert" style="float:left; margin:0 7px 20px 0;"></span> No metadata available, click OK to continue or cancel to go back to safety')
	    .dialog({
	    dialogClass: "no-close",
            modal: true,
            resizable: false,
	    title: 'Warning',
	    buttons: {
	        Continue: function () {
	            defer.resolve("true");
	    	    $(this).dialog("close");
		    run_the_NTAI_Query();
            
	    	},
	    	Cancel: function () {
	    	    defer.resolve("false");
	    	    $(this).dialog("close");
                    $('#map_di_ntai_refresh').attr('src', './js/img/refresh.png');
     	        },
	    },
	    close: function () {
		$(this).remove();
            }
	});

	$message_NO_metadata_LST_MOD11C2005.dialog('open').parent().addClass("ui-state-error");
	return defer.promise()

    };


    confirmation().then(function (answer) {
	console.log(answer);
	var ansbool = answer == "true";
	if (ansbool) {
	    console.log('continue as answer confirmation is ' + ansbool); //TRUE
	} else {
	    console.log('return as answer confirmation is ' + ansbool); //FALSE
	    return;
	}
    });    
};



function run_the_NTAI_Query() {


    var date_in_ = JSON.stringify(date_in);
    var bbox_in_ = JSON.stringify(bbox_in);

    console.log('bbox_in: ' + bbox_in_);
    console.log('bbox_in_saved_ntai: ' + bbox_in_saved_ntai);
    console.log('date_in: ' + date_in_);
    console.log('date_in_saved_ntai: ' + date_in_saved_ntai);
    console.log('image_link_ntai: ' + image_link_ntai_saved);


    console.log(date_in_ === date_in_saved_ntai);
    console.log(bbox_in_ === bbox_in_saved_ntai);
    console.log(image_link_ntai_saved);


    // If file is already computed, don't compute again, but load the saved URL
    if (date_in_ === date_in_saved_ntai && bbox_in_ === bbox_in_saved_ntai && image_link_ntai_saved){ //

        // ▾▾▾▾▾
	console.log('function within IF statement NTAI');

	var layers = map.getLayersByName("SLD test");
	for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
	map.removeLayer(layers[layerIndex]);
	};

	WMS_SLD = new OpenLayers.Layer.WMS("SLD test", image_link_ntai_saved.slice(1,-1), {
	"layers": "map",
	"format": "image/png",
	"version": "1.1.1",
	"transparent": "TRUE",
	"SLD": "http://159.226.117.95:50080/drought/settings_mapserv/sld_raster_ntai.xml"
	});
	WMS_SLD.setOpacity(0.7);
	map.addLayer(WMS_SLD);
	transform.deactivate(); //The remove the box with handles
	vectors.destroyFeatures();
	$('#map_di_ntai_refresh').attr('src', './js/img/refresh.png');
	if ($('#legend_layer')) $('#legend_layer').remove();
	$('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link_ntai_saved.slice(1,-1) +
	'&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95:50080%2Fdrought%2Fsettings_mapserv%2Fsld_raster_ntai_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
	).appendTo('body'); // CGI Legend
	if ($('#date_layer')) $('#date_layer').remove();
	$('<div id="date_layer" class="date_layer"><b>Date: ' + date_in + '</b></div>').appendTo(
	'body'); // Date


        // ▴▴▴▴▴
    // If file is not yet computed execute PyWPS service    
    } else { // ▾▾▾▾▾      
	console.log('function within ELSE statement NTAI');   
    // set the proxy
    // OpenLayers.ProxyHost = "/cgi-bin/proxy.cgi?url=";
    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecuted_NTAI_DI
    });
    // define the process and append it to OpenLayers.WPS instance
    var NTAI_SLD = new OpenLayers.WPS.Process({
        identifier: "WPS_NTAI_DI_CAL",
        inputs: [
            new OpenLayers.WPS.LiteralPut({
                identifier: "bbox",
                value: bbox_in
                    //asReference: true
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "date",
                value: date_in
            })
        ],
        async: false,
        outputs: [
            new OpenLayers.WPS.ComplexPut({
                identifier: "map",
                asReference: true
            })
        ]
    });

	date_in_saved_ntai = JSON.stringify(date_in);        
	bbox_in_saved_ntai = JSON.stringify(bbox_in); 

    // defined earlier
    wps.addProcess(NTAI_SLD);
    // run Execute
    wps.execute("WPS_NTAI_DI_CAL");
    }; // ▴▴▴▴▴
};
};
// --- // --- // --- // --- // --- // --- // --- // --- //
var WMS_SLD
    /**

 * WPS events
 */
    // Everything went OK 
onExecuted_NTAI_DI = function(process) {
    // Check if previous layer is computed, if so, remove it
    console.log('we are already here!');
    var layers = map.getLayersByName("SLD test");
    for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
        map.removeLayer(layers[layerIndex]);
    }
    image_out = process.outputs[0].getValue();
    image_out_slice = image_out.slice(43, 118);
    if (endsWith(image_out_slice, '%')) {
        console.log('image_slicing last character'),
            image_out_slice = image_out_slice.slice(0, -1)
    };
    if (endsWith(image_out_slice, '%2')) {
        console.log('image_slicing last character'),
            image_out_slice = image_out_slice.slice(0, -2)
    };
    console.log('image_out_slice: ' + image_out_slice);
    image_link = '../cgi-bin/mapserv?map='
    image_link += image_out_slice;
    image_link_ntai_saved = JSON.stringify(image_link);
    //"http://localhost/cgi-bin/mapserv?map=/var/www/html/wpsoutputs/wps23332-tmpKCoWpM.map"
    console.log('image_out_slice: ' + image_link);
    WMS_SLD = new OpenLayers.Layer.WMS("SLD test", image_link, {
        "layers": "map",
        "format": "image/png",
        "version": "1.1.1",
        "transparent": "TRUE",
        "SLD": "http://159.226.117.95:50080/drought/settings_mapserv/sld_raster_ntai.xml"
    });
    WMS_SLD.setOpacity(0.7);
    map.addLayer(WMS_SLD);

    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    $('#map_di_ntai_refresh').attr('src', './js/img/refresh.png');
    if ($('#legend_layer')) $('#legend_layer').remove();
    $('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link +
        '&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95:50080%2Fdrought%2Fsettings_mapserv%2Fsld_raster_ntai_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
    ).appendTo('body'); // CGI Legend
    if ($('#date_layer')) $('#date_layer').remove();
    $('<div id="date_layer" class="date_layer"><b>Date: ' + date_in + '</b></div>').appendTo(
        'body'); // Date
};

// --- // --- // --- // --- // ---

// NETAI // --- // --- // --- // --- // --- // --- // --- //
executeNETAIQuery = function() {
    lattop = 53.7;
    latbottom = -11.6;
    longright = 141.1;
    longleft = 73.5;
    if ($('#startDate_map').length > 0) {
        // dostuff
        date_in = $('#startDate_map').MonthPicker('GetSelectedYear') + '-' + pad($(
            '#startDate_map').MonthPicker('GetSelectedMonth')) + '-' + '01';
    } else {
        date_in = '2015-01-01';
        console.log('date_in not undefined, set to default:');
    };
    if (c) {        
        var bbox_in = owl.deepCopy(c);
        console.log('can read bbox_in from b: ' + JSON.stringify(bbox_in));        
    } else {
        //console.log(e.message);
	var bbox_in = [longleft,latbottom,longright,lattop];
        console.log('set bbox_in to default:');
    };


    var date_in_ = JSON.stringify(date_in);
    var bbox_in_ = JSON.stringify(bbox_in);

    console.log('bbox_in: ' + bbox_in_);
    console.log('bbox_in_saved_netai: ' + bbox_in_saved_netai);
    console.log('date_in: ' + date_in_);
    console.log('date_in_saved_netai: ' + date_in_saved_netai);
    console.log('image_link_netai: ' + image_link_netai_saved);


    console.log(date_in_ === date_in_saved_netai);
    console.log(bbox_in_ === bbox_in_saved_netai);
    console.log(image_link_netai_saved);


    // If file is already computed, don't compute again, but load the saved URL
    if (date_in_ === date_in_saved_netai && bbox_in_ === bbox_in_saved_netai && image_link_netai_saved){ //

        // ▾▾▾▾▾
	console.log('function within IF statement NETAI');

	var layers = map.getLayersByName("SLD test");
	for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
	map.removeLayer(layers[layerIndex]);
	};

	WMS_SLD = new OpenLayers.Layer.WMS("SLD test", image_link_netai_saved.slice(1,-1), {
	"layers": "map",
	"format": "image/png",
	"version": "1.1.1",
	"transparent": "TRUE",
	"SLD": "http://159.226.117.95:50080/drought/settings_mapserv/sld_raster_netai.xml"
	});
	WMS_SLD.setOpacity(0.7);
	map.addLayer(WMS_SLD);
	transform.deactivate(); //The remove the box with handles
	vectors.destroyFeatures();
	$('#map_di_netai_refresh').attr('src', './js/img/refresh.png');
	if ($('#legend_layer')) $('#legend_layer').remove();
	$('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link_netai_saved.slice(1,-1) +
	'&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95:50080%2Fdrought%2Fsettings_mapserv%2Fsld_raster_netai_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
	).appendTo('body'); // CGI Legend
	if ($('#date_layer')) $('#date_layer').remove();
	$('<div id="date_layer" class="date_layer"><b>Date: ' + date_in + '</b></div>').appendTo(
	'body'); // Date



        // ▴▴▴▴▴
    // If file is not yet computed execute PyWPS service    
    } else { // ▾▾▾▾▾      
	console.log('function within ELSE statement NETAI');   
    // set the proxy
    // OpenLayers.ProxyHost = "/cgi-bin/proxy.cgi?url=";
    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecuted_NETAI_DI
    });
    // define the process and append it to OpenLayers.WPS instance
    var NETAI_SLD = new OpenLayers.WPS.Process({
        identifier: "WPS_NETAI_DI_CAL",
        inputs: [
            new OpenLayers.WPS.LiteralPut({
                identifier: "bbox",
                value: bbox_in
                    //asReference: true
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "date",
                value: date_in
            })
        ],
        async: false,
        outputs: [
            new OpenLayers.WPS.ComplexPut({
                identifier: "map",
                asReference: true
            })
        ]
    });

	date_in_saved_netai = JSON.stringify(date_in);        
	bbox_in_saved_netai = JSON.stringify(bbox_in); 

    // defined earlier
    wps.addProcess(NETAI_SLD);
    // run Execute
    wps.execute("WPS_NETAI_DI_CAL");
    }; // ▴▴▴▴▴
};
// --- // --- // --- // --- // --- // --- // --- // --- //
var WMS_SLD
    /**

 * WPS events
 */
    // Everything went OK 
onExecuted_NETAI_DI = function(process) {
    // Check if previous layer is computed, if so, remove it
    console.log('we are already here!');
    var layers = map.getLayersByName("SLD test");
    for (var layerIndex = 0; layerIndex < layers.length; layerIndex++) {
        map.removeLayer(layers[layerIndex]);
    }
    image_out = process.outputs[0].getValue();
    image_out_slice = image_out.slice(43, 118);
    if (endsWith(image_out_slice, '%')) {
        console.log('image_slicing last character'),
            image_out_slice = image_out_slice.slice(0, -1)
    };
    if (endsWith(image_out_slice, '%2')) {
        console.log('image_slicing last character'),
            image_out_slice = image_out_slice.slice(0, -2)
    };
    console.log('image_out_slice: ' + image_out_slice);
    image_link = '../cgi-bin/mapserv?map='
    image_link += image_out_slice;
    image_link_netai_saved = JSON.stringify(image_link);
    //"http://localhost/cgi-bin/mapserv?map=/var/www/html/wpsoutputs/wps23332-tmpKCoWpM.map"
    console.log('image_out_slice: ' + image_link);
    WMS_SLD = new OpenLayers.Layer.WMS("SLD test", image_link, {
        "layers": "map",
        "format": "image/png",
        "version": "1.1.1",
        "transparent": "TRUE",
        "SLD": "http://159.226.117.95:50080/drought/settings_mapserv/sld_raster_ntai.xml"
    });
    WMS_SLD.setOpacity(0.7);
    map.addLayer(WMS_SLD);

    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    $('#map_di_netai_refresh').attr('src', './js/img/refresh.png');
    if ($('#legend_layer')) $('#legend_layer').remove();
    $('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link +
        '&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95:50080%2Fdrought%2Fsettings_mapserv%2Fsld_raster_ntai_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
    ).appendTo('body'); // CGI Legend
    if ($('#date_layer')) $('#date_layer').remove();
    $('<div id="date_layer" class="date_layer"><b>Date: ' + date_in + '</b></div>').appendTo(
        'body'); // Date
};

// --- // --- // --- // --- // ---




swtichBaselayer = function (name_newBaselayer) {
    var thelayer =(name_newBaselayer);
    var thenewbase = map.getLayersByName(thelayer)[0];
    map.setBaseLayer(thenewbase);
    map.baseLayer.setVisibility(true);
};

$("#ol-basemap").change(function(){
    var baseName = $(this).val();
    swtichBaselayer(baseName);
});

// --- // --- // --- // --- // ---

OpenLayers.Handler.Feature.prototype.activate = function() {
    var activated = false;
    if (OpenLayers.Handler.prototype.activate.apply(this, arguments)) {
        //this.moveLayerToTop();
        this.map.events.on({
            "removelayer": this.handleMapEvents,
            "changelayer": this.handleMapEvents,
            scope: this
        });
        activated = true;
    }
    return activated;
};

// --- // --- // --- // --- // ---

if (!OpenLayers.CANVAS_SUPPORTED) {
    var unsupported = OpenLayers.Util.getElement('unsupported');
    unsupported.innerHTML = 'Your browser does not support canvas, nothing to see here !';
}
layer = new OpenLayers.Layer.OSM('osmap', null, {
    eventListeners: {
        tileloaded: function(evt) {
            var ctx = evt.tile.getCanvasContext();
            if (ctx) {
                var imgd = ctx.getImageData(0, 0, evt.tile.size.w, evt.tile.size.h);
                var pix = imgd.data;
                for (var i = 0, n = pix.length; i < n; i += 4) {
                    pix[i] = pix[i + 1] = pix[i + 2] = (3 * pix[i] + 4 * pix[i + 1] +
                        pix[i + 2]) / 8;
                }
                ctx.putImageData(imgd, 0, 0);
                evt.tile.imgDiv.removeAttribute("crossorigin");
                evt.tile.imgDiv.src = ctx.canvas.toDataURL();
            }
        }
    }
});

arrayAerial = ["http://otile1.mqcdn.com/tiles/1.0.0/sat/${z}/${x}/${y}.jpg",
    "http://otile2.mqcdn.com/tiles/1.0.0/sat/${z}/${x}/${y}.jpg",
    "http://otile3.mqcdn.com/tiles/1.0.0/sat/${z}/${x}/${y}.jpg",
    "http://otile4.mqcdn.com/tiles/1.0.0/sat/${z}/${x}/${y}.jpg"
];

baseAerial = new OpenLayers.Layer.OSM('satellite', arrayAerial);
var options = {
    projection: new OpenLayers.Projection("EPSG:900913"),
    attribution: {
        title: ''
    },
    units: "m",
    maxResolution: 156543.0339,
    maxExtent: new OpenLayers.Bounds(-20037508.34, -20037508.34, 20037508.34, 20037508.34),
    layers: [layer, baseAerial]
};

map = new OpenLayers.Map("map", options);
//map.zoomToMaxExtent();
var fromProjection = new OpenLayers.Projection("EPSG:4326"); // transform from WGS 1984
var toProjection = new OpenLayers.Projection("EPSG:900913"); // to Spherical Mercator Projection
var extent = new OpenLayers.Bounds(69.3, -11.8, 195.7, 55.5).transform(fromProjection, toProjection);
map.zoomToExtent(extent);
//map.addControl(new OpenLayers.Control.LayerSwitcher());
//map.addControl(new OpenLayers.Control.MousePosition());
vectors = new OpenLayers.Layer.Vector("Vector Layer", {
    displayInLayerSwitcher: false
});

map.addLayer(vectors);

pixBox = new OpenLayers.Layer.Vector("pixBox Layer", {
    displayInLayerSwitcher: false
});

map.addLayer(pixBox);

markers = new OpenLayers.Layer.Markers("Markers");
map.addLayer(markers);
//markers.activate()

box = new OpenLayers.Control.DrawFeature(vectors, OpenLayers.Handler.RegularPolygon, {
    handlerOptions: {
        sides: 4,
        snapAngle: 90,
        irregular: true,
        persist: true
    }
});

box.handler.callbacks.done = endDrag_maps;
map.addControl(box);
transform = new OpenLayers.Control.TransformFeature(vectors, {
    rotate: false,
    irregular: true
});

transform.events.register("transformcomplete", transform, boxResize_maps);
map.addControl(transform);
map.addControl(box);

if (document.getElementById("bbox_drag_starter_maps")) {
    //box.activate();
    document.getElementById("bbox_drag_starter_maps").style.display = 'block';
    document.getElementById("bbox_drag_instruction_maps").style.display = 'none';
    document.getElementById("bbox_adjust_instruction_maps").style.display = 'none'
};

if (document.getElementById("bbox_drag_starter_order")) {
    //box.activate();
    document.getElementById("bbox_drag_starter_order").style.display = 'block';
    document.getElementById("bbox_drag_instruction_order").style.display = 'none';
    document.getElementById("bbox_adjust_instruction_order").style.display = 'none'
};

