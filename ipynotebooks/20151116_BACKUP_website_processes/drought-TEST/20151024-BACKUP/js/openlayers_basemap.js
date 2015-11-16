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
// Execute Metadata Query through PyWPS
function executeMetadataQuery() {
    // init the client
    wps = new OpenLayers.WPS(urlWPS, {
        onSucceeded: onExecutedMetadata
    });


    // define the process and append it to OpenLayers.WPS instance
    var Metadata_Process = new OpenLayers.WPS.Process({
        identifier: "WPS_METADATA",
        outputs: [
            new OpenLayers.WPS.LiteralPut({
                identifier: "modis_13c1_cov",
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "modis_11c2_cov",
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "trmm_3b42_coverage_1",
            }),
            new OpenLayers.WPS.LiteralPut({
                identifier: "gpcp",
            })
        ]
    });


    // defined earlier
    wps.addProcess(Metadata_Process);

    // run Execute
    wps.execute("WPS_METADATA");      

};

/**
 * WPS events
 */
// Everything went OK 
function onExecutedMetadata(process) {
    // MODIS_13_C1_COV
    var modis_13c1_cov = process.outputs[1].getValue().slice(1,-1).replace(/['"]+/g, '').replace(/\s+/g, '').split(',');
    window.metadata_modis_13c1_cov_from_date = modis_13c1_cov[0];
    window.metadata_modis_13c1_cov_to_date = modis_13c1_cov[1];
    window.metadata_modis_13c1_cov_temp_res = modis_13c1_cov[2];
    window.metadata_modis_13c1_cov_lonmin = modis_13c1_cov[3];
    window.metadata_modis_13c1_cov_latmin = modis_13c1_cov[4];
    window.metadata_modis_13c1_cov_lonmax = modis_13c1_cov[5];
    window.metadata_modis_13c1_cov_latmax = modis_13c1_cov[6];
    window.metadata_modis_13c1_cov_spat_res = modis_13c1_cov[7];

    // MODIS_11_C2_COV    
    // remove first and last character, remove the quotes, remove unnecesary spaces and split csv
    var modis_11c2_cov = process.outputs[0].getValue().slice(1,-1).replace(/['"]+/g, '').replace(/\s+/g, '').split(',');
    window.metadata_modis_11c2_cov_from_date = modis_11c2_cov[0];
    window.metadata_modis_11c2_cov_to_date = modis_11c2_cov[1];
    window.metadata_modis_11c2_cov_temp_res = modis_11c2_cov[2];
    window.metadata_modis_11c2_cov_lonmin = modis_11c2_cov[3];
    window.metadata_modis_11c2_cov_latmin = modis_11c2_cov[4];
    window.metadata_modis_11c2_cov_lonmax = modis_11c2_cov[5];
    window.metadata_modis_11c2_cov_latmax = modis_11c2_cov[6];
    window.metadata_modis_11c2_cov_spat_res = modis_11c2_cov[7];

    // trmm_3b42_coverage_1
    var trmm_3b42_coverage_1 = process.outputs[2].getValue().slice(1,-1).replace(/['"]+/g, '').replace(/\s+/g, '').split(',');
    window.metadata_trmm_3b42_coverage_1_from_date = trmm_3b42_coverage_1[0];
    window.metadata_trmm_3b42_coverage_1_to_date = trmm_3b42_coverage_1[1];
    window.metadata_trmm_3b42_coverage_1_temp_res = trmm_3b42_coverage_1[2];
    window.metadata_trmm_3b42_coverage_1_lonmin = trmm_3b42_coverage_1[3];
    window.metadata_trmm_3b42_coverage_1_latmin = trmm_3b42_coverage_1[4];
    window.metadata_trmm_3b42_coverage_1_lonmax = trmm_3b42_coverage_1[5];
    window.metadata_trmm_3b42_coverage_1_latmax = trmm_3b42_coverage_1[6];
    window.metadata_trmm_3b42_coverage_1_spat_res = trmm_3b42_coverage_1[7];

    // gpcp
    var gpcp = process.outputs[3].getValue().slice(1,-1).replace(/['"]+/g, '').replace(/\s+/g, '').split(',');
    window.metadata_gpcp_from_date = gpcp[0];
    window.metadata_gpcp_to_date = gpcp[1];
    window.metadata_gpcp_temp_res = gpcp[2];
    window.metadata_gpcp_lonmin = gpcp[3];
    window.metadata_gpcp_latmin = gpcp[4];
    window.metadata_gpcp_lonmax = gpcp[5];
    window.metadata_gpcp_latmax = gpcp[6];
    window.metadata_gpcp_spat_res = gpcp[7];
	


    console.log('function Metadata works: ');
};


var Init; (Init = function Init () {
    console.log( "ready!" );
    executeMetadataQuery()
})();

// END Execute Metadata Query
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------







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

    if (date_in > metadata_gpcp_to_date) {
        alert("The start date is after the available date! Dataset availability: "+ metadata_gpcp_from_date +' to '+ metadata_gpcp_to_date);
	$('#map_di_pap_refresh').attr('src', './js/img/refresh.png');
        //$(".modal_chart").dialog("close");
	return;
    };


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
	"SLD": "http://159.226.117.95/drought_/settings_mapserv/sld_raster_pap.xml"
	});
	WMS_SLD.setOpacity(0.7);
	map.addLayer(WMS_SLD);
	transform.deactivate(); //The remove the box with handles
	vectors.destroyFeatures();
	$('#map_di_pap_refresh').attr('src', './js/img/refresh.png');
	if ($('#legend_layer')) $('#legend_layer').remove();
	$('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link_pap_saved.slice(1,-1) +
	'&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95%2Fdrought__%2Fsettings_mapserv%2Fsld_raster_pap_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
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
    image_out_slice = image_out.slice(43, 90);
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
        "SLD": "http://159.226.117.95/drought_/settings_mapserv/sld_raster_pap.xml"
    });
    WMS_SLD.setOpacity(0.7);
    map.addLayer(WMS_SLD);
    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    $('#map_di_pap_refresh').attr('src', './js/img/refresh.png');
    if ($('#legend_layer')) $('#legend_layer').remove();
    $('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link +
        '&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95%2Fdrought_%2Fsettings_mapserv%2Fsld_raster_pap_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
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

    if (date_in > metadata_modis_11c2_cov_to_date) {
        alert("The start date is after the available date! Dataset availability: "+ metadata_modis_11c2_cov_from_date +' to '+ metadata_modis_11c2_cov_to_date);
	$('#map_di_vci_refresh').attr('src', './js/img/refresh.png');
        //$(".modal_chart").dialog("close");
	return;
    };


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
	"SLD": "http://159.226.117.95/drought_/settings_mapserv/sld_raster_vci.xml"
	});
	WMS_SLD.setOpacity(0.7);
	map.addLayer(WMS_SLD);
	transform.deactivate(); //The remove the box with handles
	vectors.destroyFeatures();
	$('#map_di_vci_refresh').attr('src', './js/img/refresh.png');
	if ($('#legend_layer')) $('#legend_layer').remove();
	$('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link_vci_saved.slice(1,-1) +
	'&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95%2Fdrought_%2Fsettings_mapserv%2Fsld_raster_vci_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
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
    image_out_slice = image_out.slice(43, 90);
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
        "SLD": "http://159.226.117.95/drought_/settings_mapserv/sld_raster_vci.xml"
    });
    WMS_SLD.setOpacity(0.7);
    map.addLayer(WMS_SLD);
    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    $('#map_di_vci_refresh').attr('src', './js/img/refresh.png');
    if ($('#legend_layer')) $('#legend_layer').remove();
    $('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link +
        '&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95%2Fdrought_%2Fsettings_mapserv%2Fsld_raster_vci_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
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

    if (date_in > metadata_modis_13c1_cov_to_date) {
        alert("The start date is after the available date! Dataset availability: "+ metadata_modis_13c1_cov_from_date +' to '+ metadata_modis_13c1_cov_to_date);
	$('#map_di_tci_refresh').attr('src', './js/img/refresh.png');
        //$(".modal_chart").dialog("close");
	return;
    };


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
	"SLD": "http://159.226.117.95/drought_/settings_mapserv/sld_raster_tci.xml"
	});
	WMS_SLD.setOpacity(0.7);
	map.addLayer(WMS_SLD);
	transform.deactivate(); //The remove the box with handles
	vectors.destroyFeatures();
	$('#map_di_tci_refresh').attr('src', './js/img/refresh.png');
	if ($('#legend_layer')) $('#legend_layer').remove();
	$('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link_tci_saved.slice(1,-1) +
	'&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95%2Fdrought_%2Fsettings_mapserv%2Fsld_raster_tci_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
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
    image_out_slice = image_out.slice(43, 90);
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
        "SLD": "http://159.226.117.95/drought_/settings_mapserv/sld_raster_tci.xml"
    });
    WMS_SLD.setOpacity(0.7);
    map.addLayer(WMS_SLD);

    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    $('#map_di_tci_refresh').attr('src', './js/img/refresh.png');
    if ($('#legend_layer')) $('#legend_layer').remove();
    $('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link +
        '&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95%2Fdrought_%2Fsettings_mapserv%2Fsld_raster_tci_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
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

    if (date_in > metadata_modis_13c1_cov_to_date) {
        alert("The start date is after the available date! Dataset availability: "+ metadata_modis_13c1_cov_from_date +' to '+ metadata_modis_13c1_cov_to_date);
	$('#map_di_vhi_refresh').attr('src', './js/img/refresh.png');
        //$(".modal_chart").dialog("close");
	return;
    };

    if (date_in > metadata_modis_11c2_cov_to_date) {
        alert("The start date is after the available date! Dataset availability: "+ metadata_modis_11c2_cov_from_date +' to '+ metadata_modis_11c2_cov_to_date);
	$('#map_di_vhi_refresh').attr('src', './js/img/refresh.png');
        //$(".modal_chart").dialog("close");
	return;
    };




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
	"SLD": "http://159.226.117.95/drought_/settings_mapserv/sld_raster_vhi.xml"
	});
	WMS_SLD.setOpacity(0.7);
	map.addLayer(WMS_SLD);
	transform.deactivate(); //The remove the box with handles
	vectors.destroyFeatures();
	$('#map_di_vhi_refresh').attr('src', './js/img/refresh.png');
	if ($('#legend_layer')) $('#legend_layer').remove();
	$('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link_vhi_saved.slice(1,-1) +
	'&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95%2Fdrought_%2Fsettings_mapserv%2Fsld_raster_vhi_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
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
    image_out_slice = image_out.slice(43, 90);
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
        "SLD": "http://159.226.117.95/drought_/settings_mapserv/sld_raster_vhi.xml"
    });
    WMS_SLD.setOpacity(0.7);
    map.addLayer(WMS_SLD);

    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    $('#map_di_vhi_refresh').attr('src', './js/img/refresh.png');
    if ($('#legend_layer')) $('#legend_layer').remove();
    $('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link +
        '&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95%2Fdrought_%2Fsettings_mapserv%2Fsld_raster_vhi_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
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



    if (date_in > metadata_modis_11c2_cov_to_date) {
        alert("The start date is after the available date! Dataset availability: "+ metadata_modis_11c2_cov_from_date +' to '+ metadata_modis_11c2_cov_to_date);
	$('#map_di_nvai_refresh').attr('src', './js/img/refresh.png');
        //$(".modal_chart").dialog("close");
	return;
    };


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
	"SLD": "http://159.226.117.95/drought_/settings_mapserv/sld_raster_nvai.xml"
	});
	WMS_SLD.setOpacity(0.7);
	map.addLayer(WMS_SLD);
	transform.deactivate(); //The remove the box with handles
	vectors.destroyFeatures();
	$('#map_di_nvai_refresh').attr('src', './js/img/refresh.png');
	if ($('#legend_layer')) $('#legend_layer').remove();
	$('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link_nvai_saved.slice(1,-1) +
	'&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95%2Fdrought_%2Fsettings_mapserv%2Fsld_raster_nvai_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
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
    image_out_slice = image_out.slice(43, 90);
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
        "SLD": "http://159.226.117.95/drought_/settings_mapserv/sld_raster_nvai.xml"
    });
    WMS_SLD.setOpacity(0.7);
    map.addLayer(WMS_SLD);

    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    $('#map_di_nvai_refresh').attr('src', './js/img/refresh.png');
    if ($('#legend_layer')) $('#legend_layer').remove();
    $('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link +
        '&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95%2Fdrought_%2Fsettings_mapserv%2Fsld_raster_nvai_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
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

    if (date_in > metadata_modis_13c1_cov_to_date) {
        alert("The start date is after the available date! Dataset availability: "+ metadata_modis_13c1_cov_from_date +' to '+ metadata_modis_13c1_cov_to_date);
	$('#map_di_ntai_refresh').attr('src', './js/img/refresh.png');
        //$(".modal_chart").dialog("close");
	return;
    };


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
	"SLD": "http://159.226.117.95/drought_/settings_mapserv/sld_raster_ntai.xml"
	});
	WMS_SLD.setOpacity(0.7);
	map.addLayer(WMS_SLD);
	transform.deactivate(); //The remove the box with handles
	vectors.destroyFeatures();
	$('#map_di_ntai_refresh').attr('src', './js/img/refresh.png');
	if ($('#legend_layer')) $('#legend_layer').remove();
	$('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link_ntai_saved.slice(1,-1) +
	'&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95%2Fdrought_%2Fsettings_mapserv%2Fsld_raster_ntai_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
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
    image_out_slice = image_out.slice(43, 90);
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
        "SLD": "http://159.226.117.95/drought_/settings_mapserv/sld_raster_ntai.xml"
    });
    WMS_SLD.setOpacity(0.7);
    map.addLayer(WMS_SLD);

    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    $('#map_di_ntai_refresh').attr('src', './js/img/refresh.png');
    if ($('#legend_layer')) $('#legend_layer').remove();
    $('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link +
        '&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95%2Fdrought_%2Fsettings_mapserv%2Fsld_raster_ntai_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
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
	"SLD": "http://159.226.117.95/drought_/settings_mapserv/sld_raster_netai.xml"
	});
	WMS_SLD.setOpacity(0.7);
	map.addLayer(WMS_SLD);
	transform.deactivate(); //The remove the box with handles
	vectors.destroyFeatures();
	$('#map_di_netai_refresh').attr('src', './js/img/refresh.png');
	if ($('#legend_layer')) $('#legend_layer').remove();
	$('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link_netai_saved.slice(1,-1) +
	'&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95%2Fdrought_%2Fsettings_mapserv%2Fsld_raster_netai_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
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
    image_out_slice = image_out.slice(43, 90);
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
        "SLD": "http://159.226.117.95/drought_/settings_mapserv/sld_raster_netai.xml"
    });
    WMS_SLD.setOpacity(0.7);
    map.addLayer(WMS_SLD);

    transform.deactivate(); //The remove the box with handles
    vectors.destroyFeatures();
    $('#map_di_netai_refresh').attr('src', './js/img/refresh.png');
    if ($('#legend_layer')) $('#legend_layer').remove();
    $('<div id="legend_layer" class="legend_sld"><b>Legend</b><br><img src=' + image_link +
        '&LAYER=map&FORMAT=image%2Fpng&VERSION=1.1.1&TRANSPARENT=false&SLD=http%3A%2F%2F159.226.117.95%2Fdrought_%2Fsettings_mapserv%2Fsld_raster_netai_legend.xml&SERVICE=WMS&REQUEST=GetLegendGraphic></img></div>'
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

