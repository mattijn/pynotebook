/*global Ext, OpenLayers, GeoExt, GeoAdmin*/ 

var mainPanel, mapPanel, permalinkPanel1, llayer="ch.swisstopo.fixpunkte-agnes";
GeoAdmin.OpenLayersImgPath = "../../Map/img/";


Ext.onReady(function() {
    GeoAdmin.webServicesUrl = GeoAdmin.protocol + '//mf-chsdi0t.bgdi.admin.ch/ltmoc';
    permalinkPanel1 = new GeoAdmin.PermalinkPanel({hidden: true, mail: true});

    mapPanel = new GeoAdmin.MapPanel({
        region: "center",
        border: false,
        width: 600,
        map: new GeoAdmin.Map(),
        tbar: ["->", {
            text: "permalink 1",
            enableToggle: true,
            toggleGroup: "export",
            allowDepress: true,
            toggleHandler: function(btn, state) {
                permalinkPanel1.setVisible(state);
            }
        }],
        items: [permalinkPanel1]
    });

    mainPanel = new Ext.Panel({
        renderTo: Ext.getBody(),
        layout: "border",
        width: 800,
        height: 400,
        items: [mapPanel]
    });


    mapPanel.map.zoomToMaxExtent();


    swipe = new OpenLayers.Control.Swipe({map: mapPanel.map});

    mapPanel.map.addControl(swipe);

    swipe.activate();

    if (mapPanel.map.getLayerByLayerName(llayer)) {
        mapPanel.map.removeLayerByName(llayer);
    }
    mapPanel.map.addLayerByName("ch.swisstopo.fixpunkte-agnes");


});
