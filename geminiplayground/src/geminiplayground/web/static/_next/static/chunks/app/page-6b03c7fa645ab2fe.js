(self.webpackChunk_N_E=self.webpackChunk_N_E||[]).push([[931],{22838:function(e,t,n){Promise.resolve().then(n.bind(n,60354)),Promise.resolve().then(n.bind(n,3932)),Promise.resolve().then(n.bind(n,3063))},75289:function(e,t,n){"use strict";n.d(t,{b:function(){return r}});let r=n(7908).Z.create({baseURL:"http://localhost:8081/api",headers:{"Content-type":"application/json"}})},60354:function(e,t,n){"use strict";n.r(t),n.d(t,{default:function(){return $}});var r=n(57437),a=n(2265);n(42162);var s=n(74503);function o(e){let{children:t}=e,[n,o]=(0,a.useState)(!1),l=s[n?"Check":"Copy"];return(0,r.jsx)("div",{className:"code-copy-btn",children:(0,r.jsx)(l,{onClick:e=>{let n=t.props.children;navigator.clipboard.writeText(n),o(!0),setTimeout(()=>{o(!1)},500)},size:18,color:n?"#0af20a":"#ddd"})})}var l=n(57742),i=n(68243);let c=(0,l.j)("inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2",{variants:{variant:{default:"border-transparent bg-primary text-primary-foreground hover:bg-primary/80",secondary:"border-transparent bg-secondary text-secondary-foreground hover:bg-secondary/80",destructive:"border-transparent bg-destructive text-destructive-foreground hover:bg-destructive/80",outline:"text-foreground"}},defaultVariants:{variant:"default"}});function d(e){let{className:t,variant:n,...a}=e;return(0,r.jsx)("div",{className:(0,i.cn)(c({variant:n}),t),...a})}var u=n(20606),f=n(26509),m=n(6752),p=n.n(m);function g(e){let{className:t}=e;return(0,r.jsxs)("div",{className:"".concat(p().loader," ").concat(t),children:[(0,r.jsx)("span",{}),(0,r.jsx)("span",{}),(0,r.jsx)("span",{})]})}var x=n(71285),h=n(72814),v=n(86384),b=n(26355),y=n(69655),j=n(96170),w=n(193),N=n(21531),_=n(20703),S=n(67445),k=n(62177);let z=e=>{let{children:t}=e;return(0,r.jsxs)("pre",{className:"blog-pre",children:[(0,r.jsx)(o,{children:t}),t]})},R=e=>{let{markdown:t}=e;return(0,r.jsx)(x.U,{className:"markdown",remarkPlugins:[h.Z,v.Z,b.Z,y.Z],rehypePlugins:[j.Z],components:{pre:z,code(e){let{node:t,inline:n,className:a="blog-code",children:s,...o}=e,l=/language-(\w+)/.exec(a||"");return!n&&l?(0,r.jsx)(w.Z,{style:N.Z,showLineNumbers:!0,language:l[1],PreTag:"div",...o,children:String(s).replace(/\n$/,"")}):(0,r.jsx)("code",{className:a,...o,children:s})}},children:t})};function Z(e){let{message:{content:t,moment:n}}=e;return(0,r.jsxs)("div",{className:"flex flex-row m-1 p-1",children:[(0,r.jsx)("div",{className:"content-center",children:(0,r.jsx)(u.Z,{className:"size-10"})}),(0,r.jsxs)("div",{className:"flex flex-col rounded-2xl bg-background fade-in ml-2 w-full p-3",children:[(0,r.jsx)("span",{className:"p-2",children:t}),(0,r.jsxs)("div",{className:"flex flex-row-re items-center gap-2 w-full place-content-end",children:[(0,r.jsx)("p",{className:"text-sm font-semibold",children:"User"}),(0,r.jsx)("p",{className:"text-xs text-muted-foreground",children:n})]})]})]})}function T(e){let{message:{content:t,moment:n,loading:a=!1}}=e;return(0,r.jsxs)("div",{className:"flex flex-row m-1 p-1",children:[(0,r.jsxs)("div",{className:"flex flex-col rounded-2xl bg-background fade-in mr-2 w-full p-3",children:[(0,r.jsx)("span",{className:"p-2",children:a?(0,r.jsx)(g,{className:"absolute"}):(0,r.jsx)(R,{markdown:t})}),(0,r.jsxs)("div",{className:"flex flex-row-re items-center gap-2 w-full place-content-end",children:[(0,r.jsx)("p",{className:"text-sm font-semibold",children:"Model"}),(0,r.jsx)("p",{className:"text-xs text-muted-foreground",children:n})]})]}),(0,r.jsx)("div",{className:"content-center",children:(0,r.jsx)(f.Z,{className:"size-10"})})]})}function C(e){let{messages:t=[],onClearChatQueue:n=null}=e,a=(0,r.jsx)("div",{className:"flex flex-col items-center justify-center h-full",children:(0,r.jsx)(_.default,{src:"gemini-logo.svg",alt:"Gemini Logo",width:200,height:200})});if(0===t.length)return a;let s=[...t].reverse().map((e,t)=>"user"===e.role?(0,r.jsx)(Z,{message:e},t):(0,r.jsx)(T,{message:e},t));return(0,r.jsxs)("div",{className:"relative pt-[40px]",children:[(0,r.jsx)(d,{variant:"outline",className:"absolute left-0 top-3",children:"Output"}),(0,r.jsxs)(S.z,{className:"absolute right-0 top-0",variant:"ghost",onClick:function(){console.log("Clearing chat history"),n&&n()},children:[(0,r.jsx)(k.XHJ,{})," clear"]}),s.length>0?s:a]})}let M=a.forwardRef((e,t)=>{let{className:n,...a}=e;return(0,r.jsx)("textarea",{className:(0,i.cn)("flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50",n),ref:t,...a})});M.displayName="Textarea",n(80763),n(29622);var I=n(85758),Y=n.n(I);function V(e){return'\n        <tag title="'.concat(e.value,"\"\n                contenteditable='false'\n                spellcheck='false'\n                tabIndex=\"-1\"\n                class='tagify__tag tagify__tag--secondary'\n                {this.getAttributes(tagData)}>\n            <x title='' class='tagify__tag__removeBtn' role='button' aria-label='remove tag'></x>\n            <div>\n                <div class='tagify__tag__avatar-wrap' >\n                    <img  alt=\"icon\" height=\"32px\" width=\"32px\" src=\"").concat(e.icon,'" />\n                </div>\n                <span class=\'tagify__tag-text\' style="margin-left: 10px">').concat(e.name,"</span>\n            </div>\n        </tag>\n        ")}function O(e){return"\n        <div ".concat(this.getAttributes(e),"\n            class='tagify__dropdown__item ").concat(e.class?e.class:"",'\'\n            tabindex="0"\n            role="option">\n            ').concat(e.icon?'\n                <div class=\'tagify__dropdown__item__avatar-wrap\'>\n                    <img onerror="this.style.visibility=\'hidden\'" height="16" src="'.concat(e.icon,'">\n                </div>'):"","\n            <strong>").concat(e.name,"</strong> <br>\n            <span>").concat(e.description,"</span>\n        </div>\n    ")}let D=(0,a.forwardRef)(function(e,t){let{settings:n,whitelist:s,loading:o,...l}=e,i=(0,a.useRef)(),c=(0,a.useRef)();return(0,a.useEffect)(()=>{if(!i.current||void 0===Y())return;let e=new(Y())(i.current,{...n,...l});return c.current=e,e.on("edit:input",e=>{console.log("edit:input",e)}),()=>{e.destroy(),c.current=null}},[i]),(0,a.useImperativeHandle)(t,()=>({currentValue:()=>c.current.getMixedTagsAsString(),clear:()=>{c.current.removeAllTags()}}),[c.current,i.current]),(0,a.useEffect)(()=>{c.current&&o&&c.current.loading(!0).dropdown.hide()},[o]),(0,a.useEffect)(()=>{let e=c.current;e&&(e.settings.whitelist.length=0,s&&s.length&&e.settings.whitelist.push(...s))},[s]),(0,r.jsx)("div",{className:"tags-input",children:(0,r.jsx)(M,{ref:i,className:"w-full shadow-none focus-visible:ring-0"})})}),A=(0,a.forwardRef)(function(e,t){let{inputTags:n,settings:a,...s}=e;return a={maxTags:20,mode:"mix",pattern:/@/,tagTextProp:"name",skipInvalid:!0,enforceWhitelist:!0,placeholder:"Type your message here...",dropdown:{enabled:0,classname:"files-list",maxItems:6,mapValueTo:"name",includeSelectedTags:!0},duplicates:!0,templates:{dropdownItemNoMatch:function(e){return"<div class='".concat(this.settings.classNames.dropdownItem,'\' value="noMatch" tabindex="0" role="option">No suggestion found for: <strong>').concat(e.value,"</strong></div>")},dropdownItem:O,tag:V},...a},(0,r.jsx)(D,{ref:t,settings:a,whitelist:n,readOnly:!1})});n(56643);var E=n(86085);function P(e){let{inputTags:t=[],tagValueAccessor:n="name",onMessageSend:s=null}=e,o=(0,a.useRef)();return(0,r.jsxs)("form",{onSubmit:e=>{e.preventDefault();let t=o.current,r=t.currentValue();r&&(r=r.replace(/\[\[(.*?)\]\]/g,e=>JSON.parse(e)[0].map(e=>"["+e[n]+"]").join(", ")),s&&(s(r),t.clear()))},className:"relative rounded-lg border bg-background focus-within:ring-1 focus-within:ring-ring p-2",children:[(0,r.jsx)(A,{ref:o,inputTags:t}),(0,r.jsx)("div",{className:"flex items-center p-3 pt-0 mt-2",children:(0,r.jsxs)(S.z,{type:"submit",size:"sm",className:"ml-auto gap-1.5",children:["Send Message",(0,r.jsx)(E.Z,{className:"size-3.5"})]})})]})}var L=n(47082),U=n(94642),B=n(20568),G=n(75289),J=n(42151),q=n.n(J);function $(){(0,L.NL)();let[e,t]=(0,a.useState)([]),n="/getTags",{data:s,isLoading:o,refetch:l}=(0,U.a)({queryKey:[n],queryFn:async()=>(await G.b.get(n)).data}),{mutate:i,data:c}=(0,B.D)({mutationFn:async e=>(await G.b.post("/generate",e)).data});(0,a.useEffect)(()=>{if(c){var e;let n=null===(e=(null==c?void 0:c.candidates)[0])||void 0===e?void 0:e.content;if(n){let e=null==n?void 0:n.parts[0];t(t=>{let n={...t[t.length-1],content:e.text,loading:!1};return[...t.slice(0,-1),n]})}else t(e=>{let t={...e[e.length-1],content:"Sorry, I couldn't understand that. Can you please rephrase?",loading:!1};return[...e.slice(0,-1),t]})}},[c]);let d=e=>{t(t=>[...t,{role:e.role,content:e.content,timestamp:new Date().toISOString(),loading:e.loading,moment:e.moment}])},u=async(e,t)=>{d({role:"user",content:e.replace("[","").replace("]",""),timestamp:new Date().toISOString(),loading:!1,moment:q()().format("MMMM Do YYYY, h:mm:ss a")}),d({role:"model",content:"Generating response...",timestamp:new Date().toISOString(),loading:!0,moment:q()().format("MMMM Do YYYY, h:mm:ss a")});let n={message:e,model:"models/gemini-1.5-pro-latest"};console.log("sending message",n),await i(n)};return(0,r.jsxs)("div",{className:"flex flex-col  rounded-xl bg-muted/50 p-1 border-2 h-screen lg:h-[calc(100vh-90px)] w-full m-0",children:[(0,r.jsx)("div",{className:"w-full h-dvh overflow-auto",children:(0,r.jsx)(C,{messages:e,onClearChatQueue:()=>{console.log("clearing chat queue"),t([])}})}),(0,r.jsx)("div",{className:"w-full",children:(0,r.jsx)(P,{inputTags:s,onMessageSend:u,tagValueAccessor:"name"})})]})}},67445:function(e,t,n){"use strict";n.d(t,{d:function(){return i},z:function(){return c}});var r=n(57437),a=n(2265),s=n(59143),o=n(57742),l=n(68243);let i=(0,o.j)("inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",{variants:{variant:{default:"bg-primary text-primary-foreground hover:bg-primary/90",destructive:"bg-destructive text-destructive-foreground hover:bg-destructive/90",outline:"border border-input bg-background hover:bg-accent hover:text-accent-foreground",secondary:"bg-secondary text-secondary-foreground hover:bg-secondary/80",ghost:"hover:bg-accent hover:text-accent-foreground",link:"text-primary underline-offset-4 hover:underline"},size:{default:"h-10 px-4 py-2",sm:"h-9 rounded-md px-3",lg:"h-11 rounded-md px-8",icon:"h-10 w-10"}},defaultVariants:{variant:"default",size:"default"}}),c=a.forwardRef((e,t)=>{let{className:n,variant:a,size:o,asChild:c=!1,...d}=e,u=c?s.g7:"button";return(0,r.jsx)(u,{className:(0,l.cn)(i({variant:a,size:o,className:n})),ref:t,...d})});c.displayName="Button"},3932:function(e,t,n){"use strict";n.r(t),n.d(t,{Label:function(){return c}});var r=n(57437),a=n(2265),s=n(24602),o=n(57742),l=n(68243);let i=(0,o.j)("text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70"),c=a.forwardRef((e,t)=>{let{className:n,...a}=e;return(0,r.jsx)(s.f,{ref:t,className:(0,l.cn)(i(),n),...a})});c.displayName=s.f.displayName},3063:function(e,t,n){"use strict";n.r(t),n.d(t,{Select:function(){return d},SelectContent:function(){return x},SelectGroup:function(){return u},SelectItem:function(){return v},SelectLabel:function(){return h},SelectScrollDownButton:function(){return g},SelectScrollUpButton:function(){return p},SelectSeparator:function(){return b},SelectTrigger:function(){return m},SelectValue:function(){return f}});var r=n(57437),a=n(2265),s=n(58161),o=n(23441),l=n(85159),i=n(80037),c=n(68243);let d=s.fC,u=s.ZA,f=s.B4,m=a.forwardRef((e,t)=>{let{className:n,children:a,...l}=e;return(0,r.jsxs)(s.xz,{ref:t,className:(0,c.cn)("flex h-10 w-full items-center justify-between rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 [&>span]:line-clamp-1",n),...l,children:[a,(0,r.jsx)(s.JO,{asChild:!0,children:(0,r.jsx)(o.Z,{className:"h-4 w-4 opacity-50"})})]})});m.displayName=s.xz.displayName;let p=a.forwardRef((e,t)=>{let{className:n,...a}=e;return(0,r.jsx)(s.u_,{ref:t,className:(0,c.cn)("flex cursor-default items-center justify-center py-1",n),...a,children:(0,r.jsx)(l.Z,{className:"h-4 w-4"})})});p.displayName=s.u_.displayName;let g=a.forwardRef((e,t)=>{let{className:n,...a}=e;return(0,r.jsx)(s.$G,{ref:t,className:(0,c.cn)("flex cursor-default items-center justify-center py-1",n),...a,children:(0,r.jsx)(o.Z,{className:"h-4 w-4"})})});g.displayName=s.$G.displayName;let x=a.forwardRef((e,t)=>{let{className:n,children:a,position:o="popper",...l}=e;return(0,r.jsx)(s.h_,{children:(0,r.jsxs)(s.VY,{ref:t,className:(0,c.cn)("relative z-50 max-h-96 min-w-[8rem] overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-md data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2","popper"===o&&"data-[side=bottom]:translate-y-1 data-[side=left]:-translate-x-1 data-[side=right]:translate-x-1 data-[side=top]:-translate-y-1",n),position:o,...l,children:[(0,r.jsx)(p,{}),(0,r.jsx)(s.l_,{className:(0,c.cn)("p-1","popper"===o&&"h-[var(--radix-select-trigger-height)] w-full min-w-[var(--radix-select-trigger-width)]"),children:a}),(0,r.jsx)(g,{})]})})});x.displayName=s.VY.displayName;let h=a.forwardRef((e,t)=>{let{className:n,...a}=e;return(0,r.jsx)(s.__,{ref:t,className:(0,c.cn)("py-1.5 pl-8 pr-2 text-sm font-semibold",n),...a})});h.displayName=s.__.displayName;let v=a.forwardRef((e,t)=>{let{className:n,children:a,...o}=e;return(0,r.jsxs)(s.ck,{ref:t,className:(0,c.cn)("relative flex w-full cursor-default select-none items-center rounded-sm py-1.5 pl-8 pr-2 text-sm outline-none focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",n),...o,children:[(0,r.jsx)("span",{className:"absolute left-2 flex h-3.5 w-3.5 items-center justify-center",children:(0,r.jsx)(s.wU,{children:(0,r.jsx)(i.Z,{className:"h-4 w-4"})})}),(0,r.jsx)(s.eT,{children:a})]})});v.displayName=s.ck.displayName;let b=a.forwardRef((e,t)=>{let{className:n,...a}=e;return(0,r.jsx)(s.Z0,{ref:t,className:(0,c.cn)("-mx-1 my-1 h-px bg-muted",n),...a})});b.displayName=s.Z0.displayName},56643:function(e,t,n){"use strict";n.d(t,{_v:function(){return d},aJ:function(){return c},pn:function(){return l},u:function(){return i}});var r=n(57437),a=n(2265),s=n(38152),o=n(68243);let l=s.zt,i=s.fC,c=s.xz,d=a.forwardRef((e,t)=>{let{className:n,sideOffset:a=4,...l}=e;return(0,r.jsx)(s.VY,{ref:t,sideOffset:a,className:(0,o.cn)("z-50 overflow-hidden rounded-md border bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md animate-in fade-in-0 zoom-in-95 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2",n),...l})});d.displayName=s.VY.displayName},68243:function(e,t,n){"use strict";n.d(t,{cn:function(){return s}});var r=n(75504),a=n(51367);function s(){for(var e=arguments.length,t=Array(e),n=0;n<e;n++)t[n]=arguments[n];return(0,a.m6)((0,r.W)(t))}},29622:function(){},42162:function(){},6752:function(e){e.exports={loader:"styles_loader__bURed"}}},function(e){e.O(0,[310,990,799,341,977,460,971,69,744],function(){return e(e.s=22838)}),_N_E=e.O()}]);