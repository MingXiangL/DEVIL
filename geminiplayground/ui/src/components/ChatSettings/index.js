"use client";
import {Input} from "@/components/ui/input";
import {useQuery, useQueryClient} from "@tanstack/react-query";
import ModelsSelect from "@/components/ModelsSelect";
import {Form, FormField, FormItem, FormLabel, FormMessage,} from "@/components/ui/form"
import {useForm} from "react-hook-form"
import {zodResolver} from "@hookform/resolvers/zod";
import {z} from "zod"
import {forwardRef, useEffect, useImperativeHandle} from "react";

const FormSchema = z.object({
    model: z.string().min(1),
    temperature: z.coerce.number().min(0.0).max(2.0),
    // candidateCount: z.coerce.number().int(),
    topP: z.coerce.number().min(0.0).max(1.0),
    topK: z.coerce.number().int()
})
const ChatSettings = forwardRef(function ChatSettings(props, ref) {

    const queryClient = useQueryClient();

    const {data: models} = useQuery({
        queryKey: ["models"],
        queryFn: async () => {
            return queryClient.getQueryData(["models"]) || [];
        }
    });

    const {data: selectedModel} = useQuery({
        queryKey: ["selectedModel"],
        enabled: models && models.length > 0,
        queryFn: async () => {
            const selectModel = queryClient.getQueryData(["selectedModel"])
            if (selectModel) {
                return selectModel;
            }
            return models[0]?.name;
        }
    });

    const form = useForm({
        resolver: zodResolver(FormSchema),
        defaultValues: {
            model: selectedModel,
            temperature: 0.0,
            //candidateCount: 1,
            topP: 0.9,
            topK: 1
        }
    });

    useEffect(() => {
        if (selectedModel) {
            form.setValue("model", selectedModel, {shouldValidate: true})
        }
    }, [selectedModel]);

    const onSubmit = async (data, evt) => {
        evt.preventDefault();
        console.log(data);

    };

    useImperativeHandle(ref, () => ({
        getSettings: () => form.getValues(),
        validateSettings: () => form.trigger(),
    }));

    const handleModelChange = (value) => {
        form.setValue("model", value, {shouldValidate: true})
        queryClient.setQueryData(["selectedModel"], value);
        const models = queryClient.getQueryData(["models"]);
        console.log(models);
        if (models) {
            const model = models.find(m => m.name === value);
            console.log(model);
            if (model) {
                form.setValue("temperature", model.temperature, {shouldValidate: true})
                form.setValue("top_p", model.top_p, {shouldValidate: true})
                form.setValue("top_k", model.top_k, {shouldValidate: true})
            }
        }

        console.log(form.getValues());
    }


    return <div className="w-full">
        <Form {...form}>
            <form onSubmit={form.handleSubmit(onSubmit)}>
                <fieldset className="grid gap-6 rounded-lg border p-4">
                    <legend className="-ml-1 px-1 text-sm font-medium">
                        Settings
                    </legend>
                    <div className="grid gap-3">
                        <FormField
                            control={form.control}
                            name="model"
                            render={({field}) => (
                                <FormItem>
                                    <FormLabel>Model</FormLabel>
                                    <ModelsSelect
                                        onValueChange={(value) => handleModelChange(value)} {...field}/>
                                    <FormMessage/>
                                </FormItem>
                            )}
                        />
                    </div>

                    {/*<div className="grid gap-3">*/}
                    {/*    <FormField*/}
                    {/*        control={form.control}*/}
                    {/*        name="candidateCount"*/}
                    {/*        render={({field}) => (*/}
                    {/*            <FormItem>*/}
                    {/*                <FormLabel>Candidate Count</FormLabel>*/}
                    {/*                <Input type="number"*/}
                    {/*                       placeholder={0.0}*/}
                    {/*                       min={0.0} step={0.1}*/}
                    {/*                       {...field}/>*/}
                    {/*                <FormMessage/>*/}
                    {/*            </FormItem>*/}
                    {/*        )}*/}
                    {/*    />*/}
                    {/*</div>*/}
                    <div className="grid gap-3">
                        <FormField
                            control={form.control}
                            name="temperature"
                            render={({field}) => (
                                <FormItem>
                                    <FormLabel>Temperature</FormLabel>
                                    <Input type="number"
                                           placeholder={0.0}
                                           min={0.0} step={0.1}

                                           {...field}/>
                                    <FormMessage/>
                                </FormItem>
                            )}
                        />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="grid gap-3">
                            <FormField
                                control={form.control}
                                name="topP"
                                render={({field}) => (
                                    <FormItem>
                                        <FormLabel>Top P</FormLabel>
                                        <Input type="number"
                                               placeholder={0.0}
                                               step={0.1}
                                               {...field}/>
                                        <FormMessage/>
                                    </FormItem>
                                )}
                            />
                        </div>
                        <div className="grid gap-3">
                            <FormField
                                control={form.control}
                                name="topK"
                                render={({field}) => (
                                    <FormItem>
                                        <FormLabel>Top K</FormLabel>
                                        <Input type="number"
                                               placeholder={0.0}
                                               min={0.0} step={1}
                                               {...field}/>
                                        <FormMessage/>
                                    </FormItem>
                                )}
                            />
                        </div>
                    </div>
                </fieldset>
                {/*<Button type="submit" className="mt-4">Save</Button>*/}
            </form>
        </Form>
    </div>;
});

export default ChatSettings;