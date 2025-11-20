/** @type {import('@remix-run/dev').AppConfig} */
module.exports = {
  ignoredRouteFiles: ["**/.*"],
  future: {
    v3_fetcherPersist: true,
    v3_relativeSplatPath: true
  },
  serverModuleFormat: "esm",
};
